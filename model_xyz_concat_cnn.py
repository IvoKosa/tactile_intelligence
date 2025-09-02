import torch
import torch.nn as nn
import torch.nn.functional as F

class xyz_linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)
    
class xyz_encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = xyz_linear()

    def forward(self, x):
        B, C, L = x.shape
        n_regions = C // 3
        x = x.view(B, n_regions, 3, L)
        x = x.permute(0, 1, 3, 2).contiguous()
        y = self.encoder(x)
        y = y.squeeze(-1)
        return y

class XYZGrouper1D_Multi(nn.Module):
    """
    Uses multiple independent xyz_encoders, one per region.
    Input:  (B, C, L), C = 3 * n_regions, channels ordered x1,y1,z1,x2,y2,z2,...
    Output: (B, n_regions, L)  (scalar per region, per time step)
    """
    def __init__(self, n_regions: int):
        super().__init__()
        self.n_regions = n_regions
        self.encoders = nn.ModuleList([xyz_linear()
                                       for _ in range(n_regions)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected (B, C, L); got {tuple(x.shape)}")
        B, C, L = x.shape
        if C != 3 * self.n_regions:
            raise ValueError(f"C must equal 3*n_regions ({3*self.n_regions}), got C={C}")

        outs = []
        for i, enc in enumerate(self.encoders):
            triplet = x[:, 3*i:3*(i+1), :].permute(0, 2, 1).contiguous()
            y = enc(triplet).squeeze(-1)
            outs.append(y)
        return torch.stack(outs, dim=1)

class Model(nn.Module):
    def __init__(self, num_features=24, mat_classes=5, tex_classes=6):
        super(Model, self).__init__()

        assert num_features % 3 == 0, 'Num Features needs to be in units of 3'
        self.num_taxels = num_features // 3

        # Single shared-weight encoder:
        # self.xyz        = xyz_encoder()

        # Multiple seperate encoders:
        self.xyz        = XYZGrouper1D_Multi(self.num_taxels)

        # Encoder
        self.conv0      = nn.Conv1d(self.num_taxels, 32, kernel_size=5, padding=2) 
        self.bn0        = nn.BatchNorm1d(32)
        self.pool0      = nn.MaxPool1d(2)

        self.conv1      = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn1        = nn.BatchNorm1d(64)
        self.pool1      = nn.MaxPool1d(2)

        self.conv2      = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2        = nn.BatchNorm1d(128)
        self.pool2      = nn.MaxPool1d(2)

        # Classification Heads
        self.flatten        = nn.Flatten()

        self.mat_fc1        = nn.LazyLinear(256)
        self.mat_dropout    = nn.Dropout(0.3)
        self.mat_fc2        = nn.Linear(256, mat_classes)

        self.tex_fc1        = nn.LazyLinear(256)
        self.tex_dropout    = nn.Dropout(0.3)
        self.tex_fc2        = nn.Linear(256, tex_classes)
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)

        # Taxel grouping
        x = self.xyz(x)

        # Conv Layers
        x = self.pool0(F.relu(self.bn0(self.conv0(x))))
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        # x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.flatten(x)
        # Cls Heads
        mat_out = self.mat_fc2(self.mat_dropout(F.relu(self.mat_fc1(x))))
        tex_out = self.tex_fc2(self.tex_dropout(F.relu(self.tex_fc1(x))))
        return mat_out, tex_out
