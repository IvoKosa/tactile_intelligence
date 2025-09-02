import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, num_features=24, mat_classes=5, tex_classes=6):
        super(Model, self).__init__()

        # -------- Encoder (unchanged from CNN) --------
        self.conv0 = nn.Conv1d(num_features, 32, kernel_size=5, padding=2)
        self.bn0   = nn.BatchNorm1d(32)
        self.pool0 = nn.MaxPool1d(2)

        self.conv1 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn1   = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)

        # -------- Classification heads --------
        self.flatten      = nn.Flatten()

        self.mat_fc1      = nn.LazyLinear(256)
        self.mat_dropout  = nn.Dropout(0.3)
        self.mat_fc2      = nn.Linear(256, mat_classes)

        self.tex_fc1      = nn.LazyLinear(256)
        self.tex_dropout  = nn.Dropout(0.3)
        self.tex_fc2      = nn.Linear(256, tex_classes)

        # -------- Decoder  --------
        self.deconv2 = nn.ConvTranspose1d(128, 64,  kernel_size=4, stride=2, padding=1)
        self.dbn2    = nn.BatchNorm1d(64)

        self.deconv1 = nn.ConvTranspose1d(64,  32,  kernel_size=4, stride=2, padding=1)
        self.dbn1    = nn.BatchNorm1d(32)

        self.deconv0 = nn.ConvTranspose1d(32,  num_features, kernel_size=4, stride=2, padding=1)
        # final layer is linear (no activation) to reconstruct continuous signals

    def encode(self, x):
        x = self.pool0(F.relu(self.bn0(self.conv0(x))))
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        return x  # (B, 128, L/8)

    def decode(self, h, target_len: int):
        y = F.relu(self.dbn2(self.deconv2(h)))
        y = F.relu(self.dbn1(self.deconv1(y)))
        y = self.deconv0(y) 
        
        # Guard: if pooling caused rounding, crop/pad to match original length
        if y.size(-1) != target_len:
            diff = target_len - y.size(-1)
            if diff > 0:
                y = F.pad(y, (diff // 2, diff - diff // 2))
            else:
                start = (-diff) // 2
                y = y[..., start:start + target_len]
        return y

    def forward(self, x, reconstruct=False):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        B, C, L = x.shape

        # ---- Encode ----
        h = self.encode(x)

        # ---- Decode  ----
        if reconstruct:
            x_hat = self.decode(h, target_len=L)
            return x_hat  # (B, C, L)

        # ---- CLS heads ----
        z = self.flatten(h)
        mat_out = self.mat_fc2(self.mat_dropout(F.relu(self.mat_fc1(z))))
        tex_out = self.tex_fc2(self.tex_dropout(F.relu(self.tex_fc1(z))))
        return mat_out, tex_out
