import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- small building blocks ----------

def _groups(c: int) -> int:
    # good default: up to 32 groups, but must divide channels
    for g in [32, 16, 8, 4, 2, 1]:
        if c % g == 0:
            return g
    return 1

class ConvBlock1D(nn.Module):
    """
    Conv -> GroupNorm -> GELU
    Optional stride>1 for downsampling. Reflection pad avoids edge artifacts.
    """
    def __init__(self, in_ch, out_ch, k=5, stride=1, dropout=0.0):
        super().__init__()
        pad = (k - 1) // 2  # 'same' receptive field with odd kernel
        self.pad = nn.ReflectionPad1d(pad)
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=stride, padding=0, bias=False)
        self.gn   = nn.GroupNorm(_groups(out_ch), out_ch)
        self.act  = nn.GELU()
        self.do   = nn.Dropout(dropout)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = self.gn(x)
        x = self.act(x)
        x = self.do(x)
        return x

class ResidualBlock1D(nn.Module):
    """
    Two ConvBlocks with a residual shortcut (1x1 if channels/stride change).
    """
    def __init__(self, in_ch, out_ch, k=3, stride=1, dropout=0.0):
        super().__init__()
        self.block1 = ConvBlock1D(in_ch, out_ch, k=k, stride=stride, dropout=dropout)
        self.block2 = ConvBlock1D(out_ch, out_ch, k=k, stride=1, dropout=dropout)
        self.proj = None
        if in_ch != out_ch or stride != 1:
            self.proj = nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        identity = x
        out = self.block1(x)
        out = self.block2(out)
        if self.proj is not None:
            identity = self.proj(identity)
        return F.gelu(out + identity)

# ---------- the model ----------

class Model(nn.Module):
    """
    Backbone with GAP and two classification heads (material & texture).
    Input: (B, 24, L)  -> variable L
    """
    def __init__(self, in_channels=24, mat_classes=10, tex_classes=20,
                 base=64, dropout=0.2, head_dropout=0.3):
        super().__init__()

        # Stem
        self.stem = nn.Sequential(
            ConvBlock1D(in_channels, base, k=7, stride=2, dropout=dropout),
            ResidualBlock1D(base, base, k=3, stride=1, dropout=dropout),
        )

        # Stages (progressively downsample; tweak depths as you like)
        self.stage1 = nn.Sequential(
            ResidualBlock1D(base, base*2, k=3, stride=2, dropout=dropout),
            ResidualBlock1D(base*2, base*2, k=3, stride=1, dropout=dropout),
        )
        self.stage2 = nn.Sequential(
            ResidualBlock1D(base*2, base*4, k=3, stride=2, dropout=dropout),
            ResidualBlock1D(base*4, base*4, k=3, stride=1, dropout=dropout),
        )

        self.out_channels = base*4

        # Global Average Pooling over the temporal axis (length dimension)
        self.gap = nn.AdaptiveAvgPool1d(1)  # -> (B, C, 1)

        # Heads
        self.head_drop = nn.Dropout(head_dropout)
        self.mat_fc = nn.Linear(self.out_channels, mat_classes)
        self.tex_fc = nn.Linear(self.out_channels, tex_classes)

        # Optional: weight init
        # self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='gelu')
        elif isinstance(m, (nn.Linear,)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        x: (B, 24, L)
        returns:
            mat_logits: (B, mat_classes)
            tex_logits: (B, tex_classes)
        """
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)

        x = self.gap(x).squeeze(-1)     # (B, C)
        x = self.head_drop(x)

        mat_logits = self.mat_fc(x)
        tex_logits = self.tex_fc(x)
        return mat_logits, tex_logits

    @torch.no_grad()
    def extract_features(self, x):
        """Return the pooled backbone features (after GAP) for analysis or a kNN probe."""
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        return self.gap(x).squeeze(-1)  # (B, C)

# ---------- example usage ----------

if __name__ == "__main__":
    B, C, L = 8, 24, 1024  # batch, channels, sequence length
    model = Model(in_channels=C, mat_classes=6, tex_classes=12)
    x = torch.randn(B, C, L)
    mat_logits, tex_logits = model(x)
    print(mat_logits.shape, tex_logits.shape)  # (8, 6) (8, 12)
