import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from eeg_preproc import EEGDataset


class SimpleNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,6,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,2)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    dset = EEGDataset("./ds003490-download", participants="participants.tsv",
                      tstart=0, tend=240, cache_amount=1, batch_size=8)

    dset_train, dset_test = dset.split(0.8, shuffle = True)
    dtrain = DataLoader(dset_train)
    dtest = DataLoader(dset_test)
    del dset
    net = SimpleNet().double()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    

    for epoch in range(2):
        
        running_loss = 0.0
        
        for i, data in enumerate(dtrain, 0):

            inputs, labels = data

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f'[{epoch+1}, {i+1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
    prtin("Finished Training")


