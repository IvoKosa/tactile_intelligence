import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(
        self,
        num_features=24,         # C
        mat_classes=5,
        tex_classes=6,
        d_model=128,             # proj size from C before LSTM
        enc_hidden=128,          # LSTM hidden per direction
        enc_layers=2,
        dec_hidden=192,          # decoder hidden size
        dec_layers=2,
        latent_dim=128,          # bottleneck
        dropout=0.2
    ):
        super().__init__()
        self.num_features = num_features

        # ---- Encoder ----
        self.in_proj   = nn.Linear(num_features, d_model)
        self.enc_lstm  = nn.LSTM(
            input_size=d_model,
            hidden_size=enc_hidden,
            num_layers=enc_layers,
            batch_first=True,
            dropout=dropout if enc_layers > 1 else 0.0,
            bidirectional=True
        )
        self.enc_norm  = nn.LayerNorm(2 * enc_hidden)
        self.to_latent = nn.Sequential(
            nn.Linear(2 * enc_hidden, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # ---- Classification heads ----
        self.mat_head = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, mat_classes)
        )
        self.tex_head = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, tex_classes)
        )

        # ---- Decoder (for reconstruction) ----
        # Map latent -> initial (h0, c0) for decoder
        self.h0_fc = nn.Linear(latent_dim, dec_layers * dec_hidden)
        self.c0_fc = nn.Linear(latent_dim, dec_layers * dec_hidden)

        # Repeat latent across time as decoder inputs (non-autoregressive)
        self.dec_lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=dec_hidden,
            num_layers=dec_layers,
            batch_first=True,
            dropout=dropout if dec_layers > 1 else 0.0,
            bidirectional=False
        )
        self.out_proj = nn.Sequential(
            nn.Linear(dec_hidden, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_features)
        )

    def encode(self, x):
        """
        x: (B, C, L) -> returns latent z: (B, latent_dim) and seq_len L
        """
        if x.dim() == 2:   # (C, L)
            x = x.unsqueeze(0)
        B, C, L = x.shape
        xt = x.transpose(1, 2)                       # (B, L, C)
        xt = self.in_proj(xt)                        # (B, L, d_model)

        enc_out, (h_n, c_n) = self.enc_lstm(xt)      # enc_out: (B, L, 2*enc_hidden)
        # Pool across time: take the last-layer outputs and mean-pool (robust)
        enc_feat = self.enc_norm(enc_out[:, :, :])   # (B, L, 2*enc_hidden)
        pooled   = enc_feat.mean(dim=1)              # (B, 2*enc_hidden)

        z = self.to_latent(pooled)                   # (B, latent_dim)
        return z, L

    def decode(self, z, L):
        """
        z: (B, latent_dim), L=sequence length
        returns reconstruction x_hat: (B, C, L)
        """
        B, D = z.shape
        # Initial states for decoder
        h0 = self.h0_fc(z).view(self.dec_lstm.num_layers, B, self.dec_lstm.hidden_size).contiguous()
        c0 = self.c0_fc(z).view(self.dec_lstm.num_layers, B, self.dec_lstm.hidden_size).contiguous()

        # Repeat latent across time as decoder input
        dec_in = z.unsqueeze(1).expand(B, L, D)      # (B, L, latent_dim)
        dec_out, _ = self.dec_lstm(dec_in, (h0, c0)) # (B, L, dec_hidden)
        y = self.out_proj(dec_out)                   # (B, L, C)
        return y.transpose(1, 2).contiguous()        # (B, C, L)

    def forward(self, x, reconstruct: bool = False):
        """
        reconstruct=False (default): returns (material_logits, texture_logits)
        reconstruct=True: returns reconstructed signal (B, C, L)
        """
        z, L = self.encode(x)
        if reconstruct:
            return self.decode(z, L)
        mat_out = self.mat_head(z)
        tex_out = self.tex_head(z)
        return mat_out, tex_out
