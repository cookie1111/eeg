import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.signal import morlet2
import sys
from tqdm import tqdm

from eeg_preproc import EEGNpDataset as EEGDataset, reshaper, transform_to_cwt, resizer
from alternative_ds import EEGCwtDataset
from time_conv import ConvTimeAttention, ConvTimeAttentionV2


def add_dim(matrix):
    mat = np.expand_dims(matrix, axis=0)
    # print(mat.shape)
    return mat

def reshaper(signals, transform=None, transform_args=None):
    if transform:
        signals = transform(signals, *transform_args)

    resa = np.reshape(signals, (512, 500))
    return resa
#lambda x,y,z: x.view(16, 512, 500)

class CoherenceClassifier(nn.Module):
    def __init__(self, num_classes, height, width):
        super(CoherenceClassifier, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
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

        self.attention = nn.MultiheadAttention(embed_dim=64, num_heads=8)

        self.fc = nn.Sequential(
            nn.Linear(31744, 1024),
            # nn.Linear(310*64,1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # Reshape x to fit the input shape requirement of nn.MultiheadAttention
        x = x.view(x.size(0), -1, 64)
        x = x.permute(1, 0, 2)  # nn.MultiheadAttention requires input shape (seq_len, batch, embed_dim)

        attn_output, _ = self.attention(x, x, x)  # Self-attention
        x = attn_output.permute(1, 0, 2)  # Reshape back to (batch, seq_len, embed_dim)

        # print(x.shape)

        # x = x.view(x.size(0), -1)  # Flatten for fully connected layer
        x = x.reshape(x.size(0), -1)  # Flatten for fully connected layer

        # print(x.shape)
        x = self.fc(x)

        return x


def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0
    correct = 0

    # Initialize progress bar
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), unit='batch')

    for batch_idx, (data, target) in progress_bar:
        data = data.view(16, 512, 500)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = output.max(1)
        correct += predicted.eq(target).sum().item()

        # Update progress bar
        progress_bar.set_description(f'Epoch: {epoch} Loss: {loss.item():.6f}')
        progress_bar.refresh()

    # Close progress bar at the end of the loop
    progress_bar.close()
    print(f"Epoch {epoch}, Loss: {running_loss / len(train_loader)}, Accuracy: {correct / len(train_loader.dataset)}")
    return running_loss / len(train_loader), correct / len(train_loader.dataset)


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

    test_loss /= len(test_loader)
    print(f"Validation set: Loss: {test_loss}, Accuracy: {correct / len(test_loader.dataset)}")
    return test_loss, correct / len(test_loader.dataset)


# Function to save the model checkpoint
def save_checkpoint(model, optimizer, epoch, checkpoint_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)


# Function to handle keyboard interrupt (Ctrl+C)
def create_signal_handler(model, optimizer):
    def signal_handler(sig, frame):
        print('Interrupted! Saving model checkpoint...')
        save_checkpoint(model, optimizer, epoch, 'interrupted_checkpoint.pth')
        print('Checkpoint saved. Exiting...')
        sys.exit(0)

    return signal_handler


def main():
    CHANNEL_WISE = False


    if CHANNEL_WISE:
        dset = EEGCwtDataset("ds003490-download", participants="participants.tsv",
                             tstart=0, tend=240, batch_size=256, debug=False, transform=add_dim)
        # Hyperparameters
        num_epochs = 10
        batch_size = 16
        learning_rate = 0.001

        dtrain, dtest = dset.split(ratios=0.8, shuffle=True, balance_classes=True)

        del dset
        for i in range(64):
            dtrain.select_channel(i)
            dtest.select_channel(i)
            # dtrain.change_mode(ch=i)
            # dtest.change_mode(ch=i)
            train_loader = torch.utils.data.DataLoader(dtrain, batch_size=batch_size, shuffle=True)
            val_loader = torch.utils.data.DataLoader(dtest, batch_size=batch_size, shuffle=False)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            num_classes = 2  # Replace with the number of classes you have
            model = CoherenceClassifier(num_classes, 120, 500).to(device)
            model = model.double()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            criterion = nn.CrossEntropyLoss()

            best_val_accuracy = 0.0
            model_save_path = f"best_model{i}.pth"

            for epoch in range(1, num_epochs + 1):
                train(model, device, train_loader, optimizer, criterion, epoch)
                val_loss, val_accuracy = test(model, device, val_loader, criterion)
                if val_accuracy >= 0.49 and val_accuracy <= 0.51:
                    print(f"{i} channel is dog shit")
                    break
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    torch.save(model.state_dict(), model_save_path)
                    print(f"Model saved at {model_save_path}")
                checkpoint_interval = 5
                if epoch % checkpoint_interval == 0:
                    save_checkpoint(model, optimizer, epoch, f'checkpoint_epoch_{epoch}.pth')
    else:
        #dset = EEGDataset("ds003490-download", participants="participants.tsv",
        #                  tstart=0, tend=240, batch_size=8, transform=reshaper,
        #                  trans_args=(transform_to_cwt,(np.linspace(1, 30, num=8), morlet2, True)))

        #dset = EEGDataset("ds003490-download", participants="participants.tsv",
        #                  tstart=0, tend=240, batch_size=64, name="_TESTER_clean",)
        #                  #transform=add_dim)  # transform=resizer, trans_args=(224,224))

        dset = EEGCwtDataset("ds003490-download", participants="participants.tsv",
                             tstart=0, tend=240, batch_size=8, )

        # Hyperparameters
        num_epochs = 50
        batch_size = 16
        learning_rate = 0.001

        # need to not transform
        dtrain, dtest = dset.split(0.8, shuffle=True)
        dtrain.select_channel(-1)
        dtest.select_channel(-1)

        train_loader = torch.utils.data.DataLoader(dtrain, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(dtest, batch_size=batch_size, shuffle=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        num_classes = 2  # Replace with the number of classes you have
        model = ConvTimeAttentionV2(num_channels=512, num_classes=2).to(device)
        #model = CoherenceClassifier(num_classes, 63, 500).to(device)
        model = model.double()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)
        criterion = nn.CrossEntropyLoss()

        best_val_accuracy = 0.0
        model_save_path = f"best_model_basic_TESTER_v2_att_cwt.pth"

        for epoch in range(1, num_epochs + 1):
            train(model, device, train_loader, optimizer, criterion, epoch)
            val_loss, val_accuracy = test(model, device, val_loader, criterion)

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save(model.state_dict(), model_save_path)
                print(f"Model saved at {model_save_path}")
            checkpoint_interval = 5
            if epoch % checkpoint_interval == 0:
                save_checkpoint(model, optimizer, epoch, f'checkpoint_epoch_{epoch}.pth')


if __name__ == "__main__":
    main()
