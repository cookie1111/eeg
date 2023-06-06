import torch
from torch import nn
import torch.nn.functional as F
import math

class Attention(nn.Module):
    def __init__(self, in_features,trans_me=True):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.tr = trans_me

    def forward(self, x):

        if self.tr:
            x = x.transpose(1, 2)  # swap dimensions
        #print(x.shape)
        weights = F.softmax(self.linear(x), dim=2)
        weights = weights.squeeze(-1).unsqueeze(1)  # change shape to (batch_size, 1, sequence_length)
        #print(weights.shape)
        weights = weights.repeat(1, x.shape[2], 1)
        #print(weights.shape)

        return weights * (x.transpose(1, 2) if self.tr else x)

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
    def __init__(self, num_channels, num_classes, expansion=1, kernel_sizes=[3,3,3,3], ff_layers = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(num_channels, 32*expansion, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32*expansion)
        self.dropout1 = nn.Dropout(0.5)
        self.attention1 = Attention(32*expansion)
        self.conv2 = nn.Conv1d(32*expansion, 64*expansion, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64*expansion)
        self.dropout2 = nn.Dropout(0.5)
        self.attention2 = Attention(64*expansion)
        self.conv3 = nn.Conv1d(64*expansion, 128*expansion, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128*expansion)
        self.dropout3 = nn.Dropout(0.5)
        self.attention3 = Attention(128*expansion)
        #self.conv4 = nn.Conv1d(128*expansion, 256*expansion, kernel_size=3, padding=1)
        #self.bn4 = nn.BatchNorm1d(256*expansion)
        #self.dropout4 = nn.Dropout(0.5)
        #self.attention4 = Attention(256*expansion)
        end_layer = 128*expansion
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
        #x = self.dropout4(F.relu(self.bn4(self.conv4(x))))
        #x = self.attention4(x)
        x = F.avg_pool1d(x, x.shape[2])
        x = x.view(x.shape[0], -1)
        for i, ff in enumerate(self.fc):
            if i < len(self.fc) - 1:  # If not the last layer, apply relu activation
                x = F.relu(ff(x))
            else:  # If the last layer, apply softmax activation
                x = ff(x)
        return x
"""
class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, batch_norm=False):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels) if batch_norm else lambda x: x
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels) if batch_norm else lambda x: x
        self.downsample = nn.Conv1d(in_channels, out_channels, 1,stride=stride) if in_channels != out_channels else None
        #self.downsample = nn.AvgPool1d(kernel_size=kernel_size,stride=stride,padding=padding) if stride != 1 else None
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out)[:,:,:residual.shape[2]])
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual[:,:,:out.shape[2]]
        out = self.relu(out)
        return out


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2, batch_norm=False):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            #print(f"Level:{i}, dilation:{dilation_size}, kernel:{kernel_size}, padding:{(kernel_size-1) * dilation_size}")
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, batch_norm=batch_norm)]

            layers += [nn.Dropout(dropout)]


        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)"""

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.bn = nn.BatchNorm1d(n_inputs)
        self.conv1 = nn.utils.weight_norm(nn.Conv2d(n_inputs, n_outputs, (1, kernel_size),
                                           stride=stride, padding=0, dilation=dilation))
        self.pad = torch.nn.ZeroPad2d((padding, 0, 0, 0))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.utils.weight_norm(nn.Conv2d(n_outputs, n_outputs, (1, kernel_size),
                                           stride=stride, padding=0, dilation=dilation))
        self.net = nn.Sequential(self.pad, self.conv1, self.relu, self.dropout,
                                 self.pad, self.conv2, self.relu, self.dropout)
        self.downsample = nn.Conv1d(
            n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        x = self.bn(x)
        out = self.net(x.unsqueeze(2)).squeeze(2)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)



class TCNClassifier(nn.Module):
    def __init__(self, num_inputs, num_channels, num_classes, num_ff, kernel_size=2, dropout=0.2, batch_norm=False, attention=False):
        super(TCNClassifier, self).__init__()
        self.tcn = TemporalConvNet(num_inputs, num_channels, kernel_size, dropout)
        self.attention = (lambda x: x) if not attention else Attention(num_channels[-1])
        self.linears = nn.ModuleList([nn.Linear(num_channels[-1]*(2**i)* (500 if i == 0 else 1) ,
                                                (num_channels[-1]*2**(i+1) if i != (num_ff-1) else num_classes)) for i in range(num_ff)])  # list of linear layers
        #self.linears = nn.ModuleList([nn.Linear(num_channels[-1]*(2**i)* 500) if i == 0 else 1) ,
        #                                         if i == (len(num_ff)-1) else num_classes)) for i,ff  in enumerate(num_ff))])  # list of linear layers

    def forward(self, x):
        # pass input through TCN layers
        x = self.tcn(x)
        # take the last value from the output sequence from the TCN
        #x = x[:, :, -1]
        x = self.attention(x)
        x = x.view(x.shape[0], -1)
        # pass through each fully connected layer with ReLU activation
        for i, linear in enumerate(self.linears):
            x = F.relu(linear(x)) if i != len(self.linears) - 1 else linear(x)  # apply ReLU only for non-last layers
        return x

