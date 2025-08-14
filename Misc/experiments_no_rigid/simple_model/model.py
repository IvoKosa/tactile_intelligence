import torch
import torch.nn as nn
import torch.nn.functional as F

class Simple_1DCNN(nn.Module):
    def __init__(self, num_features=24, num_classes=30):
        super(Simple_1DCNN, self).__init__()

        self.dual_cls   = False

        assert num_features % 3 == 0, "num_features must be divisible by 3"
        self.xyz_grouped = int(num_features/3)

        self.xyz_conv = nn.Conv1d(
            in_channels=num_features,
            out_channels=self.xyz_grouped,
            kernel_size=3,
            stride=3,
            padding=0
        )
        self.bn1    = nn.BatchNorm1d(self.xyz_grouped)
        self.drop1  = nn.Dropout(p=0.3)
        self.conv2  = nn.Conv1d(self.xyz_grouped, 64, kernel_size=3, padding=1)
        self.bn2    = nn.BatchNorm1d(64)
        self.drop2  = nn.Dropout(p=0.3)
        self.pool   = nn.AdaptiveAvgPool1d(1)
        self.fc     = nn.Linear(64, num_classes)
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        x = F.relu(self.bn1(self.xyz_conv(x)))
        x = self.drop1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.drop2(x)
        x = self.pool(x).squeeze(-1)
        x = self.fc(x) 
        return x

if __name__ =='__main__':
    input = torch.randn(1, 24, 400)

    tac_CNN = Simple_1DCNN()

    out = tac_CNN(input)

    print(out)

