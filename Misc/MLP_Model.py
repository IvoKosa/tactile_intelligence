import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMLP(nn.Module):
    def __init__(self, input_channels=24, signal_length=400, num_classes=8):
        super().__init__()
        in_feats = input_channels * signal_length  # 24 * 400 = 9600

        self.fc1 = nn.Linear(in_feats, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.drop1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.drop2 = nn.Dropout(0.5)

        self.out = nn.Linear(512, num_classes)

    def forward(self, x):
        # x: (batch, channels, length) → flatten
        x = x.view(x.size(0), -1)          # → (batch, 9600)

        x = F.relu(self.bn1(self.fc1(x)))  # → (batch, 1024)
        x = self.drop1(x)

        x = F.relu(self.bn2(self.fc2(x)))  # → (batch, 512)
        x = self.drop2(x)

        logits = self.out(x)               # → (batch, num_classes)
        return logits
