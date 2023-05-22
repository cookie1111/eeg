import torch
from torch import nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x):
        weights = F.softmax(self.linear(x), dim=1)
        return weights * x

class ConvTimeAttention(nn.Module):
    def __init__(self, num_channels, num_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(num_channels, 32, kernel_size=3, padding=1)
        self.attention1 = Attention(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.attention2 = Attention(64)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.attention1(x)
        x = F.relu(self.conv2(x))
        x = self.attention2(x)
        x = F.avg_pool1d(x, x.shape[2])
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return F.softmax(x, dim=1)

# Instantiate the model

