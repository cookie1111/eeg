#resnet input is 224*224*3 -> ResNet18

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
import torch.optim as optim

from eeg_preproc import EEGDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ResNet18(nn.Module):

    def __init__(self, num_classes, num_last_layers):
        super(ResNet18, self).__init__()
        resnet = models.resnet18(pretrained=True)

        self.conv1 = nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3,bias=False)

        self.conv1.weight= nn.Parameter(resnet.conv1.weight[:,:1,:,:])
        
        for param in resnet.parameters():
            param.requires_grad = False

        self.features = nn.Sequential(*list(resnet.children())[:-num_last_layers])
        self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1,1)),
                nn.Flatten(),
                nn.Linear(512, num_classes))

        for param in self.classifier.parameters():
            param.requiers_grad = True

    def forward(self, x):
        print(x.shape)
        x = self.features(x)
        x = self.classifier(x)

        return x

def train(model, dloader_train, optimizer, criterion,):
    a = len(dloader_train)
    i = 0
    for ins, labels in dloader_train:
        optimizer.zero_grad()

        outs = model(ins)
        loss = criterion(outs,labels)
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f"{i}/{a}")
        i = i + 1

if __name__ == "__main__":
    res = ResNet18(2,2)
    dset = EEGDataset("./ds003490-download", participants="participants.tsv",
                                  tstart=0, tend=240, cache_amount=1, batch_size=8, )
    dtrain, dtest = dset.split(ratios=0.9)
    optimizer = optim.SGD(res.parameters(), lr= 0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    train(res,DataLoader(dtest, batch_size=8,shuffle=False,num_workers=1),optimizer, criterion)

