import torch
import torch.nn as nn

class LSTM_FC_LSTM(nn.Module):
    def __init__(
        self,
        input_channels=24,
        mat_classes=6,
        tex_classes=6,
        hidden_size_1=128,
        hidden_size_2=128,
        bridge_dim=128,
        num_layers_1=2,
        num_layers_2=2,
        dropout_lstm=0.3,
        bidirectional=True,
        bridge_dropout=0.3,
    ):
        super().__init__()
        self.in_channels = input_channels
        self.mat_classes = mat_classes
        self.tex_classes = tex_classes

        # -------- LSTM #1 --------
        self.lstm1 = nn.LSTM(
            input_size=input_channels,
            hidden_size=hidden_size_1,
            num_layers=num_layers_1,
            dropout=dropout_lstm if num_layers_1 > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True
        )
        feat1 = hidden_size_1 * (2 if bidirectional else 1)

        # -------- Time-distributed FC bridge --------
        self.bridge = nn.Sequential(
            nn.Linear(feat1, bridge_dim),
            nn.ReLU(),
            nn.Dropout(bridge_dropout),
        )

        # -------- LSTM #2 --------
        self.lstm2 = nn.LSTM(
            input_size=bridge_dim,
            hidden_size=hidden_size_2,
            num_layers=num_layers_2,
            dropout=dropout_lstm if num_layers_2 > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True
        )
        feat2 = hidden_size_2 * (2 if bidirectional else 1)

        # -------- Output heads --------
        self.mat_head = nn.Sequential(
            nn.Linear(feat2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, mat_classes)
        )
        self.tex_head = nn.Sequential(
            nn.Linear(feat2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, tex_classes)
        )

    def forward(self, x):
        # Accept [B, T, F] or [B, F, T]; auto-fix if needed.
        if x.size(-1) != self.in_channels and x.size(1) == self.in_channels:
            x = x.transpose(1, 2)  # -> [B, T, F]

        # LSTM -> bridge (applies to last dim per timestep) -> LSTM
        x, _ = self.lstm1(x)      # [B, T, feat1]
        x = self.bridge(x)        # [B, T, bridge_dim]
        x, _ = self.lstm2(x)      # [B, T, feat2]

        # Global average pooling over time
        x = x.mean(dim=1)         # [B, feat2]

        # Heads
        mat_out = self.mat_head(x)
        tex_out = self.tex_head(x)
        return mat_out, tex_out


if __name__ == "__main__":
    model = LSTM_FC_LSTM()
    in_sample = torch.rand(20, 240, 24)  # [batch, seq_len, features]
    mat_out, tex_out = model(in_sample)
    print(mat_out.shape)  # torch.Size([20, 6])
    print(tex_out.shape)  # torch.Size([20, 6])
