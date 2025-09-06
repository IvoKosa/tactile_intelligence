import torch
import torch.nn as nn

class Tactile_CNN(nn.Module):
    def __init__(
        self,
        input_channels=24,
        mat_classes=6,
        tex_classes=6,
        latent_dim=64, 
        lstm_hidden=128,
        lstm_layers=1,
        bidirectional=True,
        use_last_state=False 
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.bidirectional = bidirectional
        self.use_last_state = use_last_state

        # ---------- Encoder (unchanged) ----------
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=5, stride=2, padding=2),   # (B, 64, L/2)
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),              # (B, 128, L/4)
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),             # (B, 256, L/8)
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(256, latent_dim, kernel_size=3, stride=2, padding=1),      # (B, latent_dim, L/16)
            nn.BatchNorm1d(latent_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # ---------- LSTM branch for classification ----------
        self.lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,         # input/output as (B, T, F)
            bidirectional=bidirectional
        )
        lstm_feat = lstm_hidden * (2 if bidirectional else 1)

        # Classification Heads
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

        # ---------- Decoder ----------
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, 256, kernel_size=4, stride=2, padding=1),  # (B, 256, L/8)
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),         # (B, 128, L/4)
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),          # (B, 64, L/2)
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose1d(64, input_channels, kernel_size=4, stride=2, padding=1),  # (B, 24, L)
            nn.Tanh()
        )

    def classify_from_z(self, z):
        """
        z: (B, C_latent, T_reduced)
        Returns: logits for material and texture heads
        """
        seq = z.permute(0, 2, 1).contiguous()

        out_seq, (hn, cn) = self.lstm(seq)

        if self.use_last_state:
            if self.bidirectional:
                # hn[-2] = last layer forward, hn[-1] = last layer backward
                feat = torch.cat([hn[-2], hn[-1]], dim=1)  # (B, 2 * hidden)
            else:
                feat = hn[-1]  # (B, hidden)
        else:
            # Mean-pool all time steps of the LSTM output (robust when info is spread over time)
            feat = out_seq.mean(dim=1)  # (B, hidden * num_directions)

        mat_logits = self.mat_head(feat)
        tex_logits = self.tex_head(feat)
        return mat_logits, tex_logits

    def forward(self, x, classify=True):
        z = self.encoder(x)  # (B, latent_dim, L/16)

        if classify:
            return self.classify_from_z(z)
        else:
            x_recon = self.decoder(z)
            return x_recon[..., : x.shape[-1]]
