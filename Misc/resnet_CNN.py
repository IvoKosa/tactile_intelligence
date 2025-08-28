# resnet1d.py
from typing import List, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------- Building Blocks ---------

class SE1d(nn.Module):
    """Squeeze-and-Excitation for 1D feature maps: [B, C, T] -> [B, C, T]."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(self.pool(x))
        return x * w


def same_padding(kernel_size: int, dilation: int = 1, stride: int = 1) -> int:
    """
    Returns padding that preserves T when stride=1 (PyTorch Conv1d).
    For stride > 1 we still return a reasonable padding; exact 'same'
    isn’t guaranteed but is typically fine when followed by pooling.
    """
    eff_k = dilation * (kernel_size - 1) + 1
    return (eff_k - stride) // 2


class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bn: bool = True,
        act: Optional[nn.Module] = nn.ReLU(inplace=True),
        dropout: float = 0.0,
    ):
        super().__init__()
        padding = same_padding(kernel_size, dilation, stride)
        self.conv = nn.Conv1d(
            in_ch, out_ch, kernel_size, stride=stride, padding=padding,
            dilation=dilation, groups=groups, bias=not bn
        )
        self.bn = nn.BatchNorm1d(out_ch) if bn else nn.Identity()
        self.act = act if act is not None else nn.Identity()
        self.do = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.do(x)
        return x


class ResidualBlock1d(nn.Module):
    """
    Standard ResNet block for 1D with optional bottleneck & SE.
    - If bottleneck=True, uses 1x1 -> 3x3 -> 1x1 with widths [w, w, out_ch]
    - Else, uses 3x3 -> 3x3
    """
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        bottleneck: bool = False,
        se: bool = False,
        dropout: float = 0.0,
        bn_first: bool = False,
    ):
        super().__init__()
        self.needs_proj = (in_ch != out_ch) or (stride != 1)

        if bottleneck:
            width = out_ch // 4
            self.fwd = nn.Sequential(
                ConvBNAct(in_ch, width, kernel_size=1, stride=1, dilation=1),
                ConvBNAct(width, width, kernel_size=kernel_size, stride=stride, dilation=dilation),
                ConvBNAct(width, out_ch, kernel_size=1, stride=1, dilation=1, act=None),
            )
        else:
            self.fwd = nn.Sequential(
                ConvBNAct(in_ch, out_ch, kernel_size=kernel_size, stride=stride, dilation=dilation),
                ConvBNAct(out_ch, out_ch, kernel_size=kernel_size, stride=1, dilation=dilation, act=None),
            )

        self.se = SE1d(out_ch) if se else nn.Identity()
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Projection for the skip path when shape changes
        self.proj = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm1d(out_ch),
        ) if self.needs_proj else nn.Identity()

        # Optional "pre-activation" variant (BN-ReLU before convs), for simplicity we keep post-activation as default.
        self.bn_first = bn_first
        if self.bn_first:
            self.bn1 = nn.BatchNorm1d(in_ch)

    def forward(self, x):
        identity = x
        if self.bn_first:
            x = self.bn1(x)
            x = F.relu(x, inplace=True)

        out = self.fwd(x)
        out = self.se(out)
        skip = self.proj(identity)

        # --- NEW: fix off-by-one mismatches on the time dim ---
        if out.size(-1) != skip.size(-1):
            T = min(out.size(-1), skip.size(-1))
            out = out[..., :T]
            skip = skip[..., :T]
        # ------------------------------------------------------

        out = out + skip
        out = self.act(out)
        out = self.dropout(out)
        return out


# --------- ResNet-1D Backbone ---------

class ResNet1D(nn.Module):
    """
    ResNet-style 1D backbone.
    stages: list of tuples (num_blocks, out_channels, stride, dilation)
        - stride applies to the **first block** of the stage.
        - dilation applies to all blocks in the stage (set to 1 for standard, >1 for long RF).
    Example:
        stages = [
            (2, 64, 1, 1),
            (2, 128, 2, 1),
            (2, 256, 2, 2),
            (2, 512, 2, 4),
        ]
    """
    def __init__(
        self,
        in_ch: int,
        stem_channels: int = 64,
        stem_kernel: int = 7,
        stem_stride: int = 1,
        stem_pool: Optional[str] = "max",   # 'max', 'avg', or None
        stages: List[Tuple[int, int, int, int]] = ((2, 64, 1, 1), (2, 128, 2, 1), (2, 256, 2, 2)),
        block_kernel: int = 3,
        bottleneck: bool = False,
        se: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNAct(in_ch, stem_channels, kernel_size=stem_kernel, stride=stem_stride, dilation=1),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1) if stem_pool == "max"
            else nn.AvgPool1d(kernel_size=3, stride=2, padding=1) if stem_pool == "avg"
            else nn.Identity()
        )

        curr_ch = stem_channels
        blocks = []
        for (num_blocks, out_ch, stride, dilation) in stages:
            for i in range(num_blocks):
                b_stride = stride if i == 0 else 1
                blocks.append(
                    ResidualBlock1d(
                        curr_ch, out_ch,
                        kernel_size=block_kernel,
                        stride=b_stride,
                        dilation=dilation,
                        bottleneck=bottleneck,
                        se=se,
                        dropout=dropout if (i == num_blocks - 1) else 0.0,  # light dropout at stage end
                    )
                )
                curr_ch = out_ch
        self.body = nn.Sequential(*blocks)
        self.out_dim = curr_ch

        self._init_weights()

    def forward(self, x):
        """
        x: [B, C_in, T]
        returns: [B, C_out, T_reduced]
        """
        x = self.stem(x)
        x = self.body(x)
        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


# --------- Full Model with Heads (multi-task ready) ---------

class ResNet1DClassifier(nn.Module):
    """
    Full classifier with global pooling and one or two heads.
    Useful for tactile sensing with 24 channels and two labels (e.g., texture & material).
    """
    def __init__(
        self,
        in_channels: int = 24,
        num_classes: Optional[int] = None,          # single-task
        num_classes_task1: Optional[int] = 5,    # multi-task head A
        num_classes_task2: Optional[int] = 6,    # multi-task head B
        backbone: Optional[ResNet1D] = ResNet1D(
            in_ch=24,                 # input channels
            stem_channels=64,         # first conv output channels
            stem_kernel=7,
            stem_stride=1,
            stem_pool="max",
            stages=[
                (2, 64, 1, 1),        # 2 blocks, 64 out channels, stride=1, dilation=1
                (2, 128, 2, 1),       # 2 blocks, 128 out channels, stride=2
                (2, 256, 2, 2),       # 2 blocks, 256 out channels, stride=2, dilation=2
                (2, 512, 2, 4),       # 2 blocks, 512 out channels, stride=2, dilation=4
            ],
            block_kernel=3,
            bottleneck=False,
            se=True,
            dropout=0.1
        ),
        pool: str = "avg",                          # 'avg', 'max', or 'avgmax'
        dropout: float = 0.1,
    ):
        super().__init__()
        if backbone is None:
            backbone = ResNet1D(
                in_ch=in_channels,
                stem_channels=64,
                stem_kernel=7,
                stem_stride=1,
                stem_pool="max",
                stages=[(2, 128, 1, 1), (2, 256, 2, 2), (2, 256, 2, 4)],  # decent default with dilation
                block_kernel=3,
                bottleneck=False,
                se=True,
                dropout=0.1,
            )
        self.backbone = backbone

        self.pool_type = pool
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        D = self.backbone.out_dim
        # Heads
        if (num_classes_task1 is not None) and (num_classes_task2 is not None):
            self.head1 = nn.Linear(D, num_classes_task1)
            self.head2 = nn.Linear(D, num_classes_task2)
            self.multi_task = True
        elif num_classes is not None:
            self.head = nn.Linear(D, num_classes)
            self.multi_task = False
        else:
            raise ValueError("Provide num_classes for single-task OR num_classes_task1 & num_classes_task2 for multi-task.")

        # Init heads
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.02)
                nn.init.zeros_(m.bias)

    def global_pool(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        if self.pool_type == "avg":
            x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        elif self.pool_type == "max":
            x = F.adaptive_max_pool1d(x, 1).squeeze(-1)
        elif self.pool_type == "avgmax":
            xa = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
            xm = F.adaptive_max_pool1d(x, 1).squeeze(-1)
            x = 0.5 * (xa + xm)
        else:
            raise ValueError(f"Unknown pool type: {self.pool_type}")
        return x

    def forward(self, x):
        """
        x: [B, C_in, T]
        returns:
          - single-task: logits [B, num_classes]
          - multi-task: (logits_task1, logits_task2)
        """
        x = self.backbone(x)
        x = self.global_pool(x)
        x = self.drop(x)
        if self.multi_task:
            return self.head1(x), self.head2(x)
        else:
            return self.head(x)


# --------- Utilities ---------

def receptive_field_1d(kernel_sizes: List[int], dilations: List[int], strides: List[int]) -> int:
    """
    Approximate receptive field over time for a stack of conv layers (stride/dilation aware).
    Args are lists with the same length, in layer order.
    """
    rf = 1
    acc_stride = 1
    for k, d, s in zip(kernel_sizes, dilations, strides):
        eff_k = d * (k - 1) + 1
        rf = rf + (eff_k - 1) * acc_stride
        acc_stride *= s
    return rf


# --------- Example usage / quick test ---------

if __name__ == "__main__":
    B, C, T = 8, 24, 2048   # batch, channels, time steps
    num_tex, num_mat = 12, 8

    model = ResNet1DClassifier(
        in_channels=C,
        num_classes_task1=num_tex,
        num_classes_task2=num_mat,
        pool="avgmax",
        dropout=0.1,
    )
    x = torch.randn(B, C, T)
    tex_logits, mat_logits = model(x)
    print(tex_logits.shape, mat_logits.shape)  # (8, 12) (8, 8)

    # Example loss (multi-task)
    y_tex = torch.randint(0, num_tex, (B,))
    y_mat = torch.randint(0, num_mat, (B,))
    loss = F.cross_entropy(tex_logits, y_tex) + F.cross_entropy(mat_logits, y_mat)
    loss.backward()
    print("Loss:", loss.item())
