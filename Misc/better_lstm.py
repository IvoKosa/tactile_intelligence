import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionPool(nn.Module):
    """
    Single-head additive attention pooling over time.
    Input:  (B, T, D)
    Output: (B, D)
    """
    def __init__(self, dim, hidden=128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        # x: (B, T, D)
        scores = self.proj(x).squeeze(-1)              # (B, T)
        weights = torch.softmax(scores, dim=1)         # (B, T)
        pooled = torch.bmm(weights.unsqueeze(1), x)    # (B, 1, D)
        return pooled.squeeze(1)                       # (B, D)

class DualHeadLSTM(nn.Module):
    """
    24-channel, length-848 sequence -> two classification heads (6 classes each by default)
    Pipeline: Conv1D stem -> BiLSTM (deep) -> Transformer block -> Attention Pool -> Dual Heads
    Expected input: (B, T=848, C=24)  [set transpose_input=True if input is (B, C, T)]
    """
    def __init__(
        self,
        in_channels=24,
        seq_len=848,
        lstm_hidden=256,
        lstm_layers=3,
        bidirectional=True,
        conv_channels=64,
        mha_heads=4,
        mha_ff=512,
        dropout=0.2,
        head_hidden=128,
        mat_classes=6,
        tex_classes=6
    ):
        super().__init__()
        self.transpose_needed = None  # just for debugging/consistency

        # --- 1) Temporal Conv stem (captures local patterns, reduces length a bit) ---
        # Input (B, T, C) -> we permute to (B, C, T) for convs then back.
        self.conv_stem = nn.Sequential(
            nn.Conv1d(in_channels, conv_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(inplace=True)
        )
        stem_feat = conv_channels

        # --- 2) LSTM stack ---
        self.lstm = nn.LSTM(
            input_size=stem_feat,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if lstm_layers > 1 else 0.0
        )
        lstm_feat = lstm_hidden * (2 if bidirectional else 1)

        # --- 3) Transformer encoder block (models long-range deps) ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=lstm_feat,
            nhead=mha_heads,
            dim_feedforward=mha_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # --- 4) Attention pooling over time ---
        self.attn_pool = AttentionPool(lstm_feat, hidden=128)

        # --- 5) Normalization + Heads (your enhanced heads kept) ---
        self.norm = nn.LayerNorm(lstm_feat)
        self.feat_drop = nn.Dropout(dropout)

        self.mat_head = nn.Sequential(
            nn.Linear(lstm_feat, head_hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(head_hidden, mat_classes)
        )
        self.tex_head = nn.Sequential(
            nn.Linear(lstm_feat, head_hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(head_hidden, tex_classes)
        )

        # Optional init for stability
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, transpose_input=False):
        """
        x: (B, T, C) by default; if (B, C, T) pass transpose_input=True
        """
        # if transpose_input:
        #     x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)? Actually user passes (B,C,T), so first go to (B,T,C)
        #     # Wait: if input is (B, C, T), we want conv1d with (B,C,T). So don't transpose before conv.
        #     # To keep it clear, do:
        #     x_is_BCT = True
        # else:
        #     x_is_BCT = False

        # if not x_is_BCT:
        #     # (B, T, C) -> (B, C, T) for convs
        #     x = x.transpose(1, 2)

        

        # Conv stem
        x = self.conv_stem(x)          # (B, stem_feat, T)

        # Back to (B, T, C) for LSTM/Transformer
        x = x.transpose(1, 2)          # (B, T, stem_feat)

        # LSTM
        x, _ = self.lstm(x)            # (B, T, lstm_feat)

        # Transformer block
        x = self.transformer(x)        # (B, T, lstm_feat)

        # Attention pooling
        feat = self.attn_pool(x)       # (B, lstm_feat)

        # Norm + heads
        feat = self.norm(feat)
        feat = self.feat_drop(feat)
        return self.mat_head(feat), self.tex_head(feat)
