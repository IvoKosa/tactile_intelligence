import torch
import torch.nn as nn

class Tactile_CNN(nn.Module):
    def __init__(self, input_channels=24, mat_classes=5, tex_classes=6, latent_dim=64):
        super(Tactile_CNN, self).__init__()

        # Encoder
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

        # Decoder
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
            nn.Tanh()  # adjust depending on data
        )

        # Classifier heads
        self.mat_classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),         # (B, latent_dim, 1)
            nn.Flatten(),                    # (B, latent_dim)
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

    def forward(self, x, reconstruct=False):
        z = self.encoder(x)
        if reconstruct:
            x_recon = self.decoder(z)
            return x_recon[..., :x.shape[-1]]
        else: 
            return self.mat_classifier(z), self.tex_classifier(z)
