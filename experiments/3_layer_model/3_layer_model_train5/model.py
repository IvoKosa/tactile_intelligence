import torch
import torch.nn as nn
import torch.nn.functional as F

class Tactile_CNN(nn.Module):
    def __init__(self, num_features=24, num_classes=30, input_length=850):
        super(Tactile_CNN, self).__init__()

        self.dual_cls   = False

        assert num_features % 3 == 0, "num_features must be divisible by 3"
        self.xyz_grouped = int(num_features/3)
        self.input_length = input_length

        self.xyz_conv = nn.Conv1d(
            in_channels=num_features,
            out_channels=self.xyz_grouped,
            kernel_size=3,
            stride=3,
            padding=0
        )

        self.xyz_bn     = nn.BatchNorm1d(self.xyz_grouped)

        self.conv1      = nn.Conv1d(self.xyz_grouped, 32, kernel_size=5, padding=2)
        self.bn1        = nn.BatchNorm1d(32)
        self.pool1      = nn.MaxPool1d(2)

        self.conv2      = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2        = nn.BatchNorm1d(64)
        self.pool2      = nn.MaxPool1d(2)

        self.flatten    = nn.Flatten()
        self.fc1        = nn.LazyLinear(128)
        self.dropout    = nn.Dropout(0.3)
        self.fc2        = nn.Linear(128, num_classes)

    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        x = F.relu(self.xyz_bn(self.xyz_conv(x)))
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

if __name__ =='__main__':
    input = torch.randn(1, 24, 400)

    tac_CNN = Tactile_CNN()

    out = tac_CNN(input)

    print(out)

