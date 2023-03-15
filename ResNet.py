#resnet input is 224*224*3 -> ResNet18
import copy
import time
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
import torch.optim as optim

from eeg_preproc import EEGDataset, resizer

# rewrite resnet to use th epretrained model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ResBlock(nn.Module):
    def __init__(self, input_size, output_size, stride = 1,expansion = 1, downsample = None):
        super(ResBlock, self).__init__()

        # depanding on the size of the model some have expansion from one conv layer
        # to the next within a block
        self.expansion = expansion
        self.downsample = downsample

        self.conv1 = nn.Conv2d(input_size, output_size, kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(output_size)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(output_size,output_size*expansion, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(output_size*expansion)

    def forward(self, x):
        # this puts the residual in residual network
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        oout = out + identity
        out = self.relu(out)
        return out

class ResNet18WithBlocks(nn.Module):

    def __init__(self, img_channels, num_layers, block, num_classes = 2):
        super(ResNet18WithBlocks, self).__init__()
        if num_layers == 18:

            layers = [2,2,2,2]
            self.expansion = 1

        self.in_channels = 64
        self.conv1 = nn.Conv2d(in_channels=img_channels,out_channels=self.in_channels, kernel_size = 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn. ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*self.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride = 1):
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels*self.expansion, kernel_size=1,stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, self.expansion, downsample))
        self.in_channels = out_channels * self.expansion

        for i in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, expansion=self.expansion))

        return nn.Sequential(*layers)

    def forward(self, x):
        print(x.type())
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

class ResNet18(nn.Module):

    def __init__(self, num_classes, num_last_layers):
        super(ResNet18, self).__init__()
        resnet = models.resnet18(pretrained=True)

        #self.conv1 = nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3,bias=False)

        #self.conv1.weight= nn.Parameter(resnet.conv1.weight[:,:1,:,:])

        for param in resnet.parameters():
            param.requires_grad = False

        self.features = nn.Sequential(*list(resnet.children())[:-num_last_layers])
        self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(512, num_classes))

        for param in self.classifier.parameters():
            param.requiers_grad = True

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x

def train(model, dloader_train, dloader_test, optimizer, criterion, device, num_epochs=10,channel=0):
    start = time.time()

    val_acc_hist = []

    best = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    i = 0
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs-1}')
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for ins, labels in dloader_train:
            ins = ins.to(device).double()
            labels = labels.to(device)

            optimizer.zero_grad()
            with torch.set_grad_enabled(True):

                outs = model(ins)
                loss = criterion(outs,labels)

                _, preds = torch.max(outs,1)

                loss.backward()
                optimizer.step()
            running_loss += loss.item() * ins.size(0)
            running_corrects += torch.sum(preds == labels.data)
        epoch_loss = running_loss / len(dloader_train.dataset)
        epoch_acc = running_corrects.double() / len(dloader_train.dataset)
        print(f"Train Loss: {epoch_loss} Acc: {epoch_acc}")

        model.eval()
        running_loss = 0.0
        running_corrects = 0

        for ins, labels in dloader_test:
            ins = ins.to(device).double()
            labels = labels.to(device)

            optimizer.zero_grad()
            with torch.set_grad_enabled(False):

                outs = model(ins)
                loss = criterion(outs,labels)

                _, preds = torch.max(outs,1)

            running_loss += loss.item() * ins.size(0)
            running_corrects += torch.sum(preds == labels.data)
        epoch_loss = running_loss / len(dloader_test.dataset)
        epoch_acc = running_corrects.double() / len(dloader_test.dataset)
        print(f"Test Loss: {epoch_loss} Acc: {epoch_acc}")
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
        val_acc_hist.append(epoch_acc)
    print(f"best_acc:{best_acc}")
    print(f"saving best model")
    troch.save(best_model_wts.state_dict(),f"resnet_model_channel{channel}")

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #res = ResNet18WithBlocks(1,18,ResBlock)
    dset = EEGDataset("./ds003490-download", participants="participants.tsv",
                      tstart=0, tend=240, cache_amount=1, batch_size=8,
                      transform=resizer, trans_args=(224,224))
    dtrain, dtest = dset.split(ratios=0.8, shuffle=True)
    del dset
    for i in range(64):
        res = ResNet18(2,2).to(device)
        res = res.double()
        optimizer = optim.SGD(res.parameters(), lr=0.01, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        dtrain.change_mode(ch=i)
        dtest.change_mode(ch=i)
        train(res,
              DataLoader(dtrain, batch_size=8,shuffle=False,num_workers=1),
              DataLoader(dtest, batch_size=8,shuffle=False,num_workers=1),
              optimizer, criterion, device,channel=i)

