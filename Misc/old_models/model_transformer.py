# timeseries_transformer.py
from typing import Optional, Tuple, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------- Utilities ----------------

def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """Stochastic depth per sample (when applied in main path of residual blocks)."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    # shape [B, 1, 1] to broadcast over sequence and channels
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard sinusoidal PE (no max_len cap). Creates encodings on the fly.
    Returns shape [1, L, d_model] for broadcasting.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, seq_len: int, device=None, dtype=None) -> torch.Tensor:
        device = device or torch.device("cpu")
        dtype = dtype or torch.float32
        pe = torch.zeros(seq_len, self.d_model, device=device, dtype=dtype)
        position = torch.arange(0, seq_len, device=device, dtype=dtype).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, device=device, dtype=dtype)
                             * (-math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # [1, L, D]


# ---------------- Transformer blocks ----------------

class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
        attn_dropout: float = 0.1,
        proj_dropout: float = 0.1,
        drop_path_prob: float = 0.0,
        norm_first: bool = True,
    ):
        super().__init__()
        self.norm_first = norm_first

        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=attn_dropout, batch_first=True)
        self.drop_path1 = DropPath(drop_path_prob) if drop_path_prob > 0 else nn.Identity()
        self.drop_attn = nn.Dropout(proj_dropout)

        self.ln2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(proj_dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(proj_dropout),
        )
        self.drop_path2 = DropPath(drop_path_prob) if drop_path_prob > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        if self.norm_first:
            x = x + self.drop_path1(self.drop_attn(self.attn(self.ln1(x), self.ln1(x), self.ln1(x))[0]))
            x = x + self.drop_path2(self.mlp(self.ln2(x)))
        else:
            attn_out = self.attn(x, x, x)[0]
            x = self.ln1(x + self.drop_path1(self.drop_attn(attn_out)))
            x = self.ln2(x + self.drop_path2(self.mlp(x)))
        return x


# ---------------- Main model ----------------

class TimeSeriesTransformer(nn.Module):
    """
    Time-Series Transformer for multi-channel 1D signals.

    Pipeline:
      [B,C,T] --(optional 1x1 conv channel-mixer)--> [B,C,T]
               --(Conv1d patch embed: k=patch, s=stride)--> [B,D,L]
               -> transpose to [B,L,D]
               -> add [CLS] (optional) + sinusoidal positional enc
               -> N x TransformerEncoderBlock
               -> pool ('cls' or 'mean')
               -> one or two classifier heads

    Args:
        in_channels: number of sensor channels (e.g., 24)
        num_classes: single-task classes (set either this OR both task1/task2)
        num_classes_task1/2: multi-task heads (e.g., texture/material)
        d_model: token embedding dimension
        depth: number of Transformer blocks
        n_heads: attention heads
        mlp_ratio: MLP expansion ratio
        patch_size: samples per patch along time (e.g., 16–64)
        patch_stride: stride between patches (overlap if stride < patch_size)
        attn_dropout, dropout: attention and projection dropouts
        drop_path: final DropPath probability (linearly scaled across depth)
        channel_mixer: apply a 1x1 conv over channels before patching
        pool: 'cls' (default) or 'mean'
        use_cls_token: include a learned [CLS] token

    Notes:
        - Output length L = floor((T - patch_size)/patch_stride) + 1  (no padding).
        - Ensure T >= patch_size. Variable T is fine; positions are computed on the fly.
    """
    def __init__(
        self,
        in_channels: int = 24,
        num_classes: Optional[int] = None,
        num_classes_task1: Optional[int] = None,
        num_classes_task2: Optional[int] = None,
        *,
        d_model: int = 256,
        depth: int = 6,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        patch_size: int = 32,
        patch_stride: int = 16,
        attn_dropout: float = 0.1,
        dropout: float = 0.1,
        drop_path: float = 0.1,
        channel_mixer: bool = True,
        pool: str = "cls",
        use_cls_token: bool = True,
        norm_first: bool = True,
    ):
        super().__init__()
        assert (num_classes is not None) ^ (num_classes_task1 is not None and num_classes_task2 is not None), \
            "Provide num_classes (single-task) OR both num_classes_task1 & num_classes_task2 (multi-task)."

        self.pool = pool
        self.use_cls = use_cls_token

        # Optional channel mixer (1x1 conv over channels)
        if channel_mixer:
            self.chan_mix = nn.Sequential(
                nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(in_channels),
                nn.SiLU(),
            )
        else:
            self.chan_mix = nn.Identity()

        # Patch embedding (Conv1d acts as a learned linear map from [C*patch] -> d_model)
        self.patch_embed = nn.Conv1d(
            in_channels, d_model, kernel_size=patch_size, stride=patch_stride, bias=True
        )
        self.embed_dropout = nn.Dropout(dropout)

        # Positional encoding (sinusoidal, length computed at runtime)
        self.pos_enc = SinusoidalPositionalEncoding(d_model)

        # Optional [CLS] token
        if self.use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Transformer blocks with linearly scaled DropPath
        dpr = torch.linspace(0, drop_path, depth).tolist()
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(
                d_model, n_heads,
                mlp_ratio=mlp_ratio,
                attn_dropout=attn_dropout,
                proj_dropout=dropout,
                drop_path_prob=dpr[i],
                norm_first=norm_first,
            ) for i in range(depth)
        ])
        self.norm = nn.LayerNorm(d_model)

        # Heads
        if num_classes is not None:
            self.head = nn.Linear(d_model, num_classes)
            nn.init.trunc_normal_(self.head.weight, std=0.02)
            nn.init.zeros_(self.head.bias)
            self.multi_task = False
        else:
            self.head1 = nn.Linear(d_model, num_classes_task1)
            self.head2 = nn.Linear(d_model, num_classes_task2)
            for h in [self.head1, self.head2]:
                nn.init.trunc_normal_(h.weight, std=0.02)
                nn.init.zeros_(h.bias)
            self.multi_task = True

        # Init patch embed like ViT
        nn.init.trunc_normal_(self.patch_embed.weight, std=0.02)
        if self.patch_embed.bias is not None:
            nn.init.zeros_(self.patch_embed.bias)

    @staticmethod
    def num_patches(T: int, patch_size: int, patch_stride: int) -> int:
        if T < patch_size:
            return 0
        return 1 + (T - patch_size) // patch_stride

    def forward(self, x: torch.Tensor):
        """
        x: [B, C, T]
        returns:
          - single-task: [B, num_classes]
          - multi-task: (logits_task1 [B,K1], logits_task2 [B,K2])
        """
        B, C, T = x.shape
        if T < self.patch_embed.kernel_size[0]:
            raise ValueError(f"Input length T={T} < patch_size={self.patch_embed.kernel_size[0]}")

        x = self.chan_mix(x)                          # [B, C, T]
        x = self.patch_embed(x)                       # [B, D, L]
        x = x.transpose(1, 2)                         # [B, L, D]
        L = x.size(1)

        # Add [CLS] token if used
        if self.use_cls:
            cls = self.cls_token.expand(B, -1, -1)    # [B, 1, D]
            x = torch.cat([cls, x], dim=1)            # [B, 1+L, D]

        # Positional encoding
        pos = self.pos_enc(x.size(1), device=x.device, dtype=x.dtype)  # [1, 1+L, D] or [1, L, D]
        x = x + pos
        x = self.embed_dropout(x)

        # Transformer encoder
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        # Pool
        if self.pool == "cls" and self.use_cls:
            rep = x[:, 0]                             # [B, D]
        elif self.pool == "mean":
            rep = x[:, 1:] .mean(dim=1) if self.use_cls else x.mean(dim=1)
        else:
            raise ValueError("pool must be 'cls' or 'mean' (with use_cls_token=True for 'cls').")

        # Heads
        if self.multi_task:
            return self.head2(rep), self.head1(rep)
        else:
            return self.head(rep)


# ---------------- Example / quick test ----------------
if __name__ == "__main__":
    B, C, T = 8, 24, 2048
    num_tex, num_mat = 12, 8

    # Multi-task example (texture + material)
    model = TimeSeriesTransformer(
        in_channels=C,
        num_classes_task1=num_tex,
        num_classes_task2=num_mat,
        d_model=256, depth=6, n_heads=8, mlp_ratio=4.0,
        patch_size=32, patch_stride=16,
        attn_dropout=0.1, dropout=0.1, drop_path=0.1,
        channel_mixer=True, pool="cls", use_cls_token=True
    )

    x = torch.randn(B, C, T)
    tex_logits, mat_logits = model(x)
    print(tex_logits.shape, mat_logits.shape)  # (8, 12) (8, 8)

    y_tex = torch.randint(0, num_tex, (B,))
    y_mat = torch.randint(0, num_mat, (B,))
    loss = F.cross_entropy(tex_logits, y_tex) + F.cross_entropy(mat_logits, y_mat)
    loss.backward()
    print("Loss:", loss.item())

    # Single-task example:
    single = TimeSeriesTransformer(in_channels=C, num_classes=20)
    out = single(x)
    print(out.shape)  # (8, 20)

        # self.model                      = model_transformer.TimeSeriesTransformer(
        #                                     in_channels=24,
        #                                     num_classes_task1=len(tex_classes),
        #                                     num_classes_task2=len(mat_classes),
        #                                     d_model=256, depth=6, n_heads=8, mlp_ratio=4.0,
        #                                     patch_size=32, patch_stride=16,
        #                                     attn_dropout=0.1, dropout=0.1, drop_path=0.1,
        #                                     channel_mixer=True, pool="cls", use_cls_token=True
        #                                 ).to(self.device)
