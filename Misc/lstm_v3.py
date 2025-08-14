import torch
import torch.nn as nn
import torch.nn.functional as F

class SE1D(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(channels, channels // reduction, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels // reduction, channels, 1, bias=True),
            nn.Sigmoid()
        )
    def forward(self, x):
        w = self.fc(self.pool(x))
        return x * w

class ResBlock1D(nn.Module):
    """Conv1d -> BN -> ReLU x2 with optional dilation + SE attention."""
    def __init__(self, c_in, c_out, stride=1, dilation=1, se=True):
        super().__init__()
        padding = dilation  # for kernel_size=3
        self.conv1 = nn.Conv1d(c_in, c_out, kernel_size=3, stride=stride, padding=padding, dilation=dilation)
        self.bn1   = nn.BatchNorm1d(c_out)
        self.conv2 = nn.Conv1d(c_out, c_out, kernel_size=3, stride=1, padding=padding, dilation=dilation)
        self.bn2   = nn.BatchNorm1d(c_out)
        self.se    = SE1D(c_out) if se else nn.Identity()
        self.skip  = nn.Conv1d(c_in, c_out, kernel_size=1, stride=stride) if (c_in != c_out or stride != 1) else nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = F.leaky_relu(self.bn1(self.conv1(x)), 0.2, inplace=True)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = F.leaky_relu(out + identity, 0.2, inplace=True)
        return out

class AttentionPool1D(nn.Module):
    """Learned attention over time: returns a single (B, F) vector."""
    def __init__(self, in_features, hidden=128):
        super().__init__()
        self.proj = nn.Linear(in_features, hidden)
        self.v    = nn.Linear(hidden, 1, bias=False)
    def forward(self, x):             # x: (B, T, F)
        a = torch.tanh(self.proj(x))  # (B, T, H)
        w = self.v(a).squeeze(-1)     # (B, T)
        w = torch.softmax(w, dim=1)
        pooled = torch.einsum("btf,bt->bf", x, w)
        return pooled

class Tactile_CNN(nn.Module):
    def __init__(
        self,
        input_channels=24,
        mat_classes=6,
        tex_classes=6,
        latent_dim=128,        # wider latent
        lstm_hidden=192,       # wider LSTM
        lstm_layers=2,         # deeper LSTM
        bidirectional=True,
        use_last_state=False   # kept for compatibility; we’ll prefer attention pooling
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.bidirectional = bidirectional
        self.use_last_state = use_last_state

        # -------- Encoder (residual + dilation) --------
        # Downsample by 2 at each stage: L -> L/2 -> L/4 -> L/8 -> L/16
        self.enc1 = ResBlock1D(input_channels, 64,  stride=2, dilation=1, se=True)   # (B, 64,  L/2)
        self.enc2 = ResBlock1D(64,            128, stride=2, dilation=1, se=True)   # (B, 128, L/4)
        self.enc3 = ResBlock1D(128,           256, stride=2, dilation=2, se=True)   # (B, 256, L/8)   (bigger RF)
        self.enc4 = ResBlock1D(256,           latent_dim, stride=2, dilation=4, se=True) # (B, C_lat, L/16)

        # -------- LSTM branch for classification --------
        self.lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.2 if lstm_layers > 1 else 0.0
        )
        lstm_feat = lstm_hidden * (2 if bidirectional else 1)
        self.attn_pool = AttentionPool1D(lstm_feat, hidden=128)  # learned temporal pooling

        # Heads (a bit wider)
        def head(n_out):
            return nn.Sequential(
                nn.Linear(lstm_feat, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.4),
                nn.Linear(256, n_out)
            )
        self.mat_head = head(mat_classes)
        self.tex_head = head(tex_classes)

        # -------- Decoder (U-Net style with skips) --------
        # Deconvs will upsample; we concat corresponding encoder activations
        self.up1 = nn.ConvTranspose1d(latent_dim, 256, kernel_size=4, stride=2, padding=1)  # -> L/8
        self.dec1 = ResBlock1D(256 + 256, 256, stride=1, dilation=1, se=True)

        self.up2 = nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1)        # -> L/4
        self.dec2 = ResBlock1D(128 + 128, 128, stride=1, dilation=1, se=True)

        self.up3 = nn.ConvTranspose1d(128, 64,  kernel_size=4, stride=2, padding=1)        # -> L/2
        self.dec3 = ResBlock1D(64 + 64, 64, stride=1, dilation=1, se=True)

        self.up4 = nn.ConvTranspose1d(64, input_channels, kernel_size=4, stride=2, padding=1)  # -> L
        self.out_act = nn.Tanh()

    def encode(self, x):
        f1 = self.enc1(x)
        f2 = self.enc2(f1)
        f3 = self.enc3(f2)
        z  = self.enc4(f3)    # (B, C_latent, L/16)
        return z, (f1, f2, f3)

    def decode(self, z, skips):
        f1, f2, f3 = skips
        u1 = self.up1(z)                        # (B, 256, L/8)
        d1 = self.dec1(torch.cat([u1, f3], 1))  # (B, 256, L/8)

        u2 = self.up2(d1)                       # (B, 128, L/4)
        d2 = self.dec2(torch.cat([u2, f2], 1))  # (B, 128, L/4)

        u3 = self.up3(d2)                       # (B, 64, L/2)
        d3 = self.dec3(torch.cat([u3, f1], 1))  # (B, 64, L/2)

        out = self.up4(d3)                      # (B, C_in, L)
        return self.out_act(out)

    def classify_from_z(self, z):
        # Prepare sequence for LSTM: (B, T_reduced, C_latent)
        seq = z.permute(0, 2, 1).contiguous()
        out_seq, (hn, cn) = self.lstm(seq)  # (B, T, F)

        if self.use_last_state:
            if self.bidirectional:
                feat = torch.cat([hn[-2], hn[-1]], dim=1)
            else:
                feat = hn[-1]
        else:
            feat = self.attn_pool(out_seq)  # learned attention over time

        mat_logits = self.mat_head(feat)
        tex_logits = self.tex_head(feat)
        return mat_logits, tex_logits

    def forward(self, x, classify=False):
        """
        x: (B, C_in, L)
        If classify=False: returns reconstruction (trimmed to input length).
        If classify=True:  returns (mat_logits, tex_logits).
        """
        z, skips = self.encode(x)
        if classify:
            return self.classify_from_z(z)
        x_recon = self.decode(z, skips)
        return x_recon[..., : x.shape[-1]]
