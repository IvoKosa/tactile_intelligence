import torch
import torch.nn as nn

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

        # -------------------- Encoder --------------------
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(2),  # L -> L/2

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(2),  # L/2 -> L/4

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(2)  # L/4 -> L/8
        )

        # -------------------- LSTM Module --------------------

        # LSTM PARAMS
        hidden_size     = 128
        num_layers      = 2
        lstm_dropout    = 0.3
        bidirectional   = True

        self.lstm = nn.LSTM(
            input_size      = 128,
            hidden_size     = hidden_size,
            num_layers      = num_layers,
            dropout         = lstm_dropout,
            bidirectional   = bidirectional,
            batch_first     = True
        )

        lstm_feat = hidden_size * (2 if bidirectional else 1)

        # -------------------- Classification Heads --------------------
        self.mat_head = nn.Sequential(
            nn.Linear(lstm_feat, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, mat_classes)
        )
        self.tex_head = nn.Sequential(
            nn.Linear(lstm_feat, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, tex_classes)
        )

    def forward(self, x):
        # X.shape: [B, C, L]
        x = self.encoder(x)
        x = x.transpose(1, 2)
        x, (h_n, c_n) = self.lstm(x)
        x = x.transpose(1, 2)
        x = torch.nn.functional.adaptive_avg_pool1d(x, 1).squeeze(-1)
        mat_out = self.mat_head(x)
        tex_out = self.tex_head(x)
        return mat_out, tex_out

if __name__ == '__main__':

    model = Model()
    in_sample = torch.rand([32, 24, 240])
    mat_out, tex_out = model(in_sample)
    print(mat_out.shape)
    print(tex_out.shape)
