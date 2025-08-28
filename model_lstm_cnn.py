import torch
import torch.nn as nn

class LSTM_CNN(nn.Module):
    def __init__(
        self,
        input_channels=24,
        mat_classes=6,
        tex_classes=6
    ):
        super().__init__()
        self.in_channels = input_channels
        self.mat_classes = mat_classes
        self.tex_classes = tex_classes

        # -------------------- LSTM Module --------------------

        # LSTM PARAMS
        hidden_size     = 128
        num_layers      = 2
        lstm_dropout    = 0.3
        bidirectional   = True

        self.lstm = nn.LSTM(
            input_size      = self.in_channels,
            hidden_size     = hidden_size,
            num_layers      = num_layers,
            dropout         = lstm_dropout,
            bidirectional   = bidirectional
        )

        lstm_feat = hidden_size * (2 if bidirectional else 1)

        # -------------------- Encoder --------------------
        self.encoder = nn.Sequential(
        nn.Conv1d(lstm_feat, 256, kernel_size=5, stride=2, padding=2), 
        nn.BatchNorm1d(256),
        nn.LeakyReLU(0.2, inplace=True),

        nn.Conv1d(256, 384, kernel_size=5, stride=2, padding=2), 
        nn.BatchNorm1d(384),
        nn.LeakyReLU(0.2, inplace=True),

        nn.Conv1d(384, 512, kernel_size=3, stride=2, padding=1), 
        nn.BatchNorm1d(512),
        nn.LeakyReLU(0.2, inplace=True),

        nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1), 
        nn.BatchNorm1d(512),
        nn.LeakyReLU(0.2, inplace=True)
        )

        # -------------------- Classification Heads --------------------
        self.mat_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, mat_classes)
        )
        self.tex_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, tex_classes)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x, (h_n, c_n) = self.lstm(x)
        x = x.transpose(1, 2)
        x = self.encoder(x)
        x = torch.nn.functional.adaptive_avg_pool1d(x, 1).squeeze(-1)
        mat_out = self.mat_head(x)
        tex_out = self.tex_head(x)
        return mat_out, tex_out

if __name__ == '__main__':

    model = LSTM_CNN()
    in_sample = torch.rand([20, 240, 24])
    mat_out, tex_out = model(in_sample)
    print(mat_out.shape)
    print(tex_out.shape)
