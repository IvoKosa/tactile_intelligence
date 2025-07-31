import torch
import torch.nn as nn
import torch.nn.functional as F

class Tactile_CNN(nn.Module):
    def __init__(self, num_features=24, mat_classes=6, tex_classes=6):
        super(Tactile_CNN, self).__init__()

        # Encoder
        self.conv0      = nn.Conv1d(num_features, 32, kernel_size=5, padding=2)
        self.bn0        = nn.BatchNorm1d(32)
        self.pool0      = nn.MaxPool1d(2)

        self.conv1      = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn1        = nn.BatchNorm1d(64)
        self.pool1      = nn.MaxPool1d(2)

        self.conv2      = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2        = nn.BatchNorm1d(128)
        self.pool2      = nn.MaxPool1d(2)

        # Latent Representation
        self.flatten        = nn.Flatten()

        self.mat_fc1        = nn.LazyLinear(256)
        self.mat_dropout    = nn.Dropout(0.3)
        self.mat_fc2        = nn.Linear(256, mat_classes)

        self.tex_fc1        = nn.LazyLinear(256)
        self.tex_dropout    = nn.Dropout(0.3)
        self.tex_fc2        = nn.Linear(256, tex_classes)

        # Decoder
