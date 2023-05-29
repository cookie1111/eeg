import torch
from torch import nn
import torch.nn.functional as F
import math

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
    def __init__(self, num_channels, num_classes, ff_layers = 1):
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
        end_layer = 128
        self.fc = nn.ModuleList()
        for i in range(ff_layers):
            print(end_layer * (2**i))
            self.fc.append(nn.Linear(end_layer * (2**i), end_layer * (2**(i+1) if  i < ff_layers-1 else num_classes)))
        #self.fc = nn.Linear

    def forward(self, x):
        x = self.dropout1(F.relu(self.bn1(self.conv1(x))))
        x = self.attention1(x)
        x = self.dropout2(F.relu(self.bn2(self.conv2(x))))
        x = self.attention2(x)
        x = self.dropout3(F.relu(self.bn3(self.conv3(x))))
        x = self.attention3(x)
        x = F.avg_pool1d(x, x.shape[2])
        x = x.view(x.shape[0], -1)
        for i, ff in enumerate(self.fc):
            if i < len(self.fc) - 1:  # If not the last layer, apply relu activation
                x = F.relu(ff(x))
            else:  # If the last layer, apply softmax activation
                x = ff(x)
        return x


import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)[:,:,:residual.shape[2]]
        if self.downsample is not None:
            residual = self.downsample(x)
        #print(residual.shape, out.shape)
        out += residual
        out = self.relu(out)
        return out


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            #print(f"Level:{i}, dilation:{dilation_size}, kernel:{kernel_size}, padding:{(kernel_size-1) * dilation_size}")
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size)]

            layers += [nn.Dropout(dropout)]


        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCNClassifier(nn.Module):
    def __init__(self, num_inputs, num_channels, num_classes, num_ff, kernel_size=2, dropout=0.2):
        super(TCNClassifier, self).__init__()
        self.tcn = TemporalConvNet(num_inputs, num_channels, kernel_size, dropout)
        self.linears = nn.ModuleList([nn.Linear(num_channels[-1]*(2**i) ,
                                                num_channels[-1]*(2**(i+1) if i == (num_ff-1) else num_classes)) for i in range(num_ff)])  # list of linear layers

    def forward(self, x):
        # pass input through TCN layers
        x = self.tcn(x)
        # take the last value from the output sequence from the TCN
        x = x[:, :, -1]
        # pass through each fully connected layer with ReLU activation
        for i, linear in enumerate(self.linears):
            x = F.relu(linear(x)) if i != len(self.linears) - 1 else linear(x)  # apply ReLU only for non-last layers
        return x

