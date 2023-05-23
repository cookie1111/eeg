import torch
from torch import nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x):

        x = x.transpose(1, 2)  # swap dimensions
        #print(x.shape)
        weights = F.softmax(self.linear(x), dim=2)
        weights = weights.squeeze(-1).unsqueeze(1)  # change shape to (batch_size, 1, sequence_length)
        #print(weights.shape)
        weights = weights.repeat(1, x.shape[2], 1)
        #print(weights.shape)

        return weights * x.transpose(1, 2)

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

class ConvTimeAttentionV2(nn.Module):
    def __init__(self, num_channels, num_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(num_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.dropout1 = nn.Dropout(0.5)
        self.attention1 = Attention(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.5)
        self.attention2 = Attention(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(0.5)
        self.attention3 = Attention(128)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.dropout1(F.relu(self.bn1(self.conv1(x))))
        x = self.attention1(x)
        x = self.dropout2(F.relu(self.bn2(self.conv2(x))))
        x = self.attention2(x)
        x = self.dropout3(F.relu(self.bn3(self.conv3(x))))
        x = self.attention3(x)
        x = F.avg_pool1d(x, x.shape[2])
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return F.softmax(x, dim=1)
