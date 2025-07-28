import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------------ Individual grouping x, y, z
class GroupLinear(nn.Module):
    def __init__(self, in_features=24, group_size=3):
        super().__init__()
        assert in_features % group_size == 0, "in_features must be divisible by group_size"
        self.num_groups = in_features // group_size
        
        # create one small Linear(3→1) for each group
        self.linears = nn.ModuleList([
            nn.Linear(group_size, 1)
            for _ in range(self.num_groups)
        ])
        
        self.activations = nn.ModuleList([nn.ReLU() for _ in range(self.num_groups)])

    def forward(self, x):
        # x: [batch_size, 24]
        # split into 8 chunks of size 3 along dim=1
        chunks = x.chunk(self.num_groups, dim=1)  # returns a tuple of 8 tensors [batch,3]
        
        # apply each little linear + activation
        outs = []
        for lin, act, chunk in zip(self.linears, self.activations, chunks):
            # lin(chunk) -> [batch,1]; act(...) -> [batch,1]
            outs.append(act(lin(chunk)))
        
        # concatenate back to [batch_size, 8]
        return torch.cat(outs, dim=1)

# ------------------------------------------------------------------ 1D CNN Shared Weights

class Tactile_CNN(nn.Module):
    def __init__(self, num_features=12, num_classes=8):
        super(Tactile_CNN, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.pool = nn.AdaptiveAvgPool1d(1)  # Reduces sequence length to 1

        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: (batch_size, time_steps, num_features)
        x = x.permute(0, 2, 1)  # → (batch_size, num_features, time_steps)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1)  # → (batch_size, 64)
        x = self.fc(x)  # → (batch_size, num_classes)
        return x
    

if __name__ =='__main__':

    batch = torch.randn(5, 24)
    m = GroupLinear(in_features=24, group_size=3)
    out = m(batch)    # out.shape == [5,8]
