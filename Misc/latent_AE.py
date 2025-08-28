import torch
import torch.nn as nn
import torch.nn.functional as F

class Tactile_CNN(nn.Module):
    def __init__(self, input_channels=24, mat_classes=5, tex_classes=6,
                 latent_dim=64, latent_fc_dim=128):
        super(Tactile_CNN, self).__init__()
        self.latent_dim = latent_dim

        # ----- Encoder -----
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(256, latent_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(latent_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # ----- Latent "FC" bottleneck (channel-wise MLP via 1x1 convs) -----
        # Acts like fully connected across channels for each time step, keeps length L' unchanged.
        self.latent_mlp = nn.Sequential(
            nn.Conv1d(latent_dim, latent_fc_dim, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(latent_fc_dim, latent_dim, kernel_size=1),
            nn.ReLU()
        )

        # ----- Decoder -----
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose1d(64, input_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

        # ----- Classifier heads (pool over time, then dense) -----
        self.mat_classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),                    # -> (B, latent_dim)
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, mat_classes)
        )
        self.tex_classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, tex_classes)
        )

    def forward(self, x, classify=False):
        z = self.encoder(x)             # (B, latent_dim, L')
        z = self.latent_mlp(z)          # (B, latent_dim, L')  -- length preserved

        if classify:
            return self.mat_classifier(z), self.tex_classifier(z)
        else:
            x_recon = self.decoder(z)   # (B, input_channels, ~L)
            return x_recon[..., :x.shape[-1]]  # trim in case of off-by-one when L not divisible by 16
