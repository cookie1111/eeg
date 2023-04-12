#resnet input is 224*224*3 -> ResNet18
import copy
import time
import torch
import torch.nn as nn
import torchvision.models as models
from scipy.signal import morlet2
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np

from eeg_preproc import EEGNpDataset as EEGDataset, resizer, transform_to_cwt, reshaper

# rewrite resnet to use th epretrained model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ResNet18(nn.Module):

    def __init__(self, num_classes, num_last_layers):
        super(ResNet18, self).__init__()
        resnet = models.resnet18(pretrained=True)

        #self.conv1 = nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3,bias=False)

        #self.conv1.weight= nn.Parameter(resnet.conv1.weight[:,:1,:,:])
        
        for param in resnet.parameters():
            param.requires_grad = False

        num_total_layers = len(list(resnet.children()))
        for i, child in enumerate(resnet.children()):
            if num_total_layers - i <= num_last_layers:
                for param in child.parameters():
                    param.requires_grad = True

        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(512, 1024),
                nn.Linear(1024, num_classes)

        )

        for param in self.classifier.parameters():
            param.requiers_grad = True

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x

def train(model, dloader_train, dloader_test, optimizer, criterion, device, num_epochs=10,channel=0,name=f"resnet_model_channel_2_layers_cwt_reshape"):
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
    torch.save(best_model_wts,name)

TEST = 1
special = False
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not special:
        dset = EEGDataset("ds003490-download", participants="participants.tsv",
                          tstart=0, tend=240, batch_size=32,
                          transform=resizer, trans_args=(224,224))
    else:
        dset = EEGDataset("ds003490-download", participants="participants.tsv",
                          tstart=0, tend=240, batch_size=8, transform=resizer,
                          trans_args=(224,
                                      224,
                                      reshaper,
                                      (transform_to_cwt, (np.linspace(1, 30, num=21), morlet2, True))),
                          debug=False)
    dtrain, dtest = dset.split(ratios=0.8, shuffle=True, balance_classes=True)
    del dset
    if TEST == 0:
        for i in range(64):
            res = ResNet18(2,2).to(device)
            res = res.double()
            optimizer = optim.SGD(res.parameters(), lr=0.01, momentum=0.9)
            criterion = nn.CrossEntropyLoss()
            dtrain.change_mode(ch=i)
            dtest.change_mode(ch=i)
            train(res,
                  DataLoader(dtrain, batch_size=32,shuffle=True,num_workers=4),
                  DataLoader(dtest, batch_size=32,shuffle=True,num_workers=4),
                  optimizer, criterion, device,channel=i)
    elif TEST == 1:
        res = ResNet18(2, 2).to(device)
        res = res.double()
        optimizer = optim.SGD(res.parameters(), lr=0.01, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        train(res,
              DataLoader(dtrain, batch_size=4, shuffle=True, num_workers=1),
              DataLoader(dtest, batch_size=4, shuffle=True, num_workers=1),
              optimizer, criterion, device, name="resnet_2unfrozen_2additional_layers")
