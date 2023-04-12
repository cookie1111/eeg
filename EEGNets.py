import torch
import torch.nn as nn
import torch.optim as optim
from eeg_preproc import EEGNpDataset as EEGDataset

class CoherenceClassifier(nn.Module):
    def __init__(self, num_classes):
        super(CoherenceClassifier, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 28 * 28, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0
    correct = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = output.max(1)
        correct += predicted.eq(target).sum().item()

    print(f"Epoch {epoch}, Loss: {running_loss / len(train_loader)}, Accuracy: {correct / len(train_loader.dataset)}")

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            correct += predicted.eq(target).sum().item()

    test_loss /= len
    test_loss /= len(test_loader)
    print(f"Validation set: Loss: {test_loss}, Accuracy: {correct / len(test_loader.dataset)}")

def main():
    dset = EEGDataset("ds003490-download", participants="participants.tsv",
                      tstart=0, tend=240, batch_size=32)
    # Hyperparameters
    num_epochs = 50
    batch_size = 16
    learning_rate = 0.001

    # Prepare your data
    train_rgb_coherence_matrices = ...  # Replace with your training RGB coherence matrices
    train_labels = ...  # Replace with your training labels
    val_rgb_coherence_matrices = ...  # Replace with your validation RGB coherence matrices
    val_labels = ...  # Replace with your validation labels

    dtrain, dtest = dset.split(ratios=0.8, shuffle=True, balance_classes=True)

    train_loader = torch.utils.data.DataLoader(dtrain, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dtest, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = 2  # Replace with the number of classes you have
    model = CoherenceClassifier(num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, num_epochs + 1):
        train(model, device, train_loader, optimizer, criterion, epoch)
        test(model, device, val_loader, criterion)

if __name__ == "__main__":
    main()
