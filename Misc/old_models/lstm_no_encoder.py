import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(
        self,
        input_channels=24,
        mat_classes=5,
        tex_classes=6
    ):
        super().__init__()
        self.in_channels = input_channels
        self.mat_classes = mat_classes
        self.tex_classes = tex_classes

        # -------------------- LSTM Module --------------------
        hidden_size   = 128
        num_layers    = 2
        lstm_dropout  = 0.3
        bidirectional = True

        self.lstm = nn.LSTM(
            input_size    = self.in_channels,   # C
            hidden_size   = hidden_size,
            num_layers    = num_layers,
            dropout       = lstm_dropout,       # between layers
            bidirectional = bidirectional,
            batch_first   = True
        )
        lstm_feat = hidden_size * (2 if bidirectional else 1)  # 256 with bi-LSTM

        # -------------------- Attention Pooling --------------------
        # Produces a scalar score per timestep, then softmax over time.
        self.attn = nn.Sequential(
            nn.Linear(lstm_feat, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        # Optional projection to 128-d to match the original heads’ input size
        self.proj = nn.Linear(lstm_feat, 128)

        # -------------------- Classification Heads --------------------
        self.mat_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, mat_classes)
        )
        self.tex_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, tex_classes)
        )

    def forward(self, x):
        # x: [B, C, L]  (e.g., L = 240)
        x = x.transpose(1, 2)                    # [B, L, C]
        x, _ = self.lstm(x)                      # [B, L, lstm_feat]

        # Attention weights over time
        scores = self.attn(x)                    # [B, L, 1]
        alpha  = F.softmax(scores, dim=1)        # [B, L, 1]

        # Weighted sum (context vector)
        ctx = torch.sum(alpha * x, dim=1)        # [B, lstm_feat]

        # Project to 128-d then classify
        z = self.proj(ctx)                       # [B, 128]
        mat_out = self.mat_head(z)               # [B, mat_classes]
        tex_out = self.tex_head(z)               # [B, tex_classes]
        return mat_out, tex_out
