import torch
import torch.nn as nn
import torch.nn.functional as F

# --- CNN encoder/decoder for a single timestep (length=24) --------------------

class Conv1dEncoder(nn.Module):
    """
    Input:  (B*T, 1, 24)
    Output: (B*T, latent_dim)
    """
    def __init__(self, latent_dim: int = 64):
        super().__init__()
        # Keep it small since length=24 is short
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1),  # 24 -> 12
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),  # 12 -> 6
            nn.ReLU(inplace=True),
        )
        # 64 channels * length 6 = 384
        self.to_latent = nn.Linear(64 * 6, latent_dim)

    def forward(self, x):
        # x: (B*T, 1, 24)
        h = self.features(x)
        h = h.flatten(1)             # (B*T, 64*6)
        z = self.to_latent(h)        # (B*T, latent_dim)
        return z


class Conv1dDecoder(nn.Module):
    """
    Input:  (B*T, latent_dim)
    Output: (B*T, 1, 24)
    """
    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.from_latent = nn.Linear(latent_dim, 64 * 6)
        # Mirror of encoder using ConvTranspose1d to go 6 -> 12 -> 24
        self.deconv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),  # 6 -> 12
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),  # 12 -> 24
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 1, kernel_size=3, padding=1),
        )

    def forward(self, z):
        h = self.from_latent(z)      # (B*T, 64*6)
        h = h.view(h.size(0), 64, 6) # (B*T, 64, 6)
        x_hat = self.deconv(h)       # (B*T, 1, 24)
        return x_hat
    
# --- Full model ---------------------------------------------------------------

class Tactile_CNN(nn.Module):
    """
    Input:  x of shape (B, T, 24)
    Returns:
      material_logits: (B, n_material_classes)
      texture_logits:  (B, n_texture_classes)
      recon:           (B, T, 24)  reconstruction per timestep
      latents:         (B, T, latent_dim)
    """
    def __init__(
        self,
        latent_dim: int = 64,
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
        bidirectional: bool = False,
        n_material_classes: int = 5,
        n_texture_classes: int = 6,
        dropout: float = 0.0,
        use_last_timestep: bool = True,  # if False, mean-pool over time
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.use_last_timestep = use_last_timestep

        self.encoder = Conv1dEncoder(latent_dim=latent_dim)
        self.decoder = Conv1dDecoder(latent_dim=latent_dim)

        self.lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        out_feat = lstm_hidden * (2 if bidirectional else 1)

        self.classifier_material = nn.Sequential(
            nn.Linear(out_feat, out_feat),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(out_feat, n_material_classes),
        )
        self.classifier_texture = nn.Sequential(
            nn.Linear(out_feat, out_feat),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(out_feat, n_texture_classes),
        )

    def forward(self, x, classify=False):
        """
        x: (B, T, 24)
        """
        x = x.permute(0, 2, 1)
        B, T, L = x.shape
        assert L == 24, f"Expected last dim=24, got {L}"

        # --- AE encode each timestep ---
        x_reshaped = x.reshape(B * T, 1, 24)        # (B*T, 1, 24)
        z = self.encoder(x_reshaped)                # (B*T, latent_dim)

        # --- AE decode for reconstruction loss --- 
        x_hat = self.decoder(z)                     # (B*T, 1, 24)

        # reshape back
        latents = z.view(B, T, -1)                  # (B, T, latent_dim)
        recon = x_hat.view(B, T, 24)                # (B, T, 24)

        # --- LSTM over time on latents ---
        lstm_out, (h_n, c_n) = self.lstm(latents)   # lstm_out: (B, T, H[*2])

        if self.use_last_timestep:
            # Take the last timestep’s output for classification
            cls_feat = lstm_out[:, -1, :]           # (B, H[*2])
        else:
            # Mean-pool over time
            cls_feat = lstm_out.mean(dim=1)         # (B, H[*2])

        material_logits = self.classifier_material(cls_feat)  # (B, C_m)
        texture_logits  = self.classifier_texture(cls_feat)   # (B, C_t)

        if classify:
            return material_logits, texture_logits
        else:
            recon = recon.permute(0, 2, 1)
            return recon