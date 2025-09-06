import torch
import torch.nn as nn

class Tactile_CNN(nn.Module):
    def __init__(self, input_channels=24, mat_classes=6, tex_classes=6, latent_dim=64):
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

    def forward(self, x, classify=False):
        z = self.encoder(x)
        if classify:
            return self.mat_classifier(z), self.tex_classifier(z)
        else: 
            x_recon = self.decoder(z)
            return x_recon[..., :x.shape[-1]]


# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class Tactile_CNN(nn.Module):
#     def __init__(self, num_features=24, mat_classes=6, tex_classes=6):
#         super(Tactile_CNN, self).__init__()

#         # Encoder
#         self.conv0      = nn.Conv1d(num_features, 32, kernel_size=5, padding=2)
#         self.bn0        = nn.BatchNorm1d(32)
#         self.pool0      = nn.MaxPool1d(2)

#         self.conv1      = nn.Conv1d(32, 64, kernel_size=5, padding=2)
#         self.bn1        = nn.BatchNorm1d(64)
#         self.pool1      = nn.MaxPool1d(2)

#         self.conv2      = nn.Conv1d(64, 128, kernel_size=3, padding=1)
#         self.bn2        = nn.BatchNorm1d(128)
#         self.pool2      = nn.MaxPool1d(2)

#         # Decoder
#         self.upool2     = nn.Upsample(scale_factor=2, mode='nearest')
#         self.deconv2    = nn.Conv1d(128, 64, kernel_size=3, padding=1)
#         self.bn_d2      = nn.BatchNorm1d(64)

#         self.upool1     = nn.Upsample(scale_factor=2, mode='nearest')
#         self.deconv1    = nn.Conv1d(64, 32, kernel_size=5, padding=2)
#         self.bn_d1      = nn.BatchNorm1d(32)

#         self.upool0     = nn.Upsample(scale_factor=2, mode='nearest')
#         self.deconv0    = nn.Conv1d(32, num_features, kernel_size=5, padding=2)
        
#         # Latent Representation
#         self.flatten        = nn.Flatten()

#         self.mat_fc1        = nn.LazyLinear(256)
#         self.mat_dropout    = nn.Dropout(0.3)
#         self.mat_fc2        = nn.Linear(256, mat_classes)

#         self.tex_fc1        = nn.LazyLinear(256)
#         self.tex_dropout    = nn.Dropout(0.3)
#         self.tex_fc2        = nn.Linear(256, tex_classes)

#     def forward(self, x):
#         if x.dim() == 2:
#             x = x.unsqueeze(0)
#         # Conv Layers
#         x = self.pool0(F.relu(self.bn0(self.conv0(x))))
#         x = self.pool1(F.relu(self.bn1(self.conv1(x))))
#         x = self.pool2(F.relu(self.bn2(self.conv2(x))))
#         x = self.flatten(x)

#         # if self.eval():
#         #     mat_out = self.mat_fc2(self.mat_dropout(F.relu(self.mat_fc1(x))))
#         #     tex_out = self.tex_fc2(self.tex_dropout(F.relu(self.tex_fc1(x))))
#         #     return mat_out, tex_out
#         # else:
#         x = self.upool2(self.deconv2(self.bn_d2(x)))
#         x = self.upool1(self.deconv1(self.bn_d1(x)))
#         x = self.upool0(self.deconv0(x))
#         x = torch.tanh(x)
#         return x