import random
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.signal import morlet2
import sys
from tqdm import tqdm
from time import sleep

from eeg_preproc import EEGNpDataset as EEGDataset, reshaper, transform_to_cwt, resizer
from alternative_ds import EEGCwtDataset
from time_conv import ConvTimeAttention, ConvTimeAttentionReduction, ConvTimeAttentionV2, ConvTimeAttentionV3, TCNClassifier



def add_dim(matrix):
    mat = np.expand_dims(matrix, axis=0)
    # print(mat.shape)
    return mat


def reshaper(signals, transform=None, transform_args=None):
    if transform:
        signals = transform(signals, *transform_args)

    resa = np.reshape(signals, (512, 500))
    return resa


# lambda x,y,z: x.view(16, 512, 500)

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
        # print(data.shape)
        # data = data.view(16, 512, 500)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        #print(output,target)
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




def test2(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            test_loss += criterion(output, target).item()
            #print(output.shape, target.shape)
            _, predicted = output.max(1)
            correct += predicted.eq(target).sum().item()

            # Store all targets and predictions for metrics computation
            all_targets.extend(target.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    test_loss /= len(test_loader)
    accuracy = correct / len(test_loader.dataset)
    print(f"Validation set: Loss: {test_loss}, Accuracy: {accuracy}")
    print(len(all_targets),len(all_predictions),all_targets[0],all_predictions[0])
    # Compute sensitivity, specificity and AUC-ROC
    tn, fp, fn, tp = confusion_matrix(all_targets, all_predictions).ravel()
    print(tn, fp, fn, tp)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    auc_roc = roc_auc_score(all_targets, all_predictions)

    print(f"Sensitivity: {sensitivity}")
    print(f"Specificity: {specificity}")
    print(f"AUC-ROC: {auc_roc}")

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(all_targets, all_predictions)
    """plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc_roc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    #plt.show()
    """
    return test_loss, accuracy

def test3(model, device, test_loaders, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    all_targets = []
    all_predictions = []
    predics = 0

    with torch.no_grad():
        for (i,test_loader) in enumerate(test_loaders):
            #print(i,test_loader)
            predics = 0
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                #print(output.shape,target.shape)
                test_loss += criterion(output, target).item()
                _, predicted = output.max(1)
                #spremenimo da se razreda zamenjata
                predicted -= 1
                predicted *= -1

                correct += predicted.eq(target).sum().item()
                predics += sum(predicted)
                # Store all targets and predictions for metrics computation
            print(predics)
            #zamenjamo target razred
            print(target)
            all_targets.append(1 if target[0] == 0 else 0)
            all_predictions.append(predics.item()/len(test_loader.dataset)) #povprečno napovedovanje

    print(all_predictions)
    print(all_targets)

    dif = 0
    fpr, tpr, thresh = roc_curve(all_targets, all_predictions)
    for i, pair in enumerate(zip(fpr,tpr)):
        print(i)
        print(pair)
        if abs(pair[1] - pair[0]) > dif:
            dif = abs(pair[1] - pair[0])
            idx = i
    thresh = thresh[idx]



    auc_roc = roc_auc_score(all_targets, all_predictions)

    print(auc_roc)

    test_loss /= sum([len(test_loader) for test_loader in test_loaders])
    accuracy = correct / sum([len(test_loader.dataset) for test_loader in test_loaders])
    #print(f"Validation set: Loss: {test_loss}, Accuracy: {accuracy}")
    #print(len(all_targets),len(all_predictions),all_targets[0],all_predictions[0])
    # Compute sensitivity, specificity and AUC-ROC

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc_roc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    # Threshold the actual results
    all_preds = [0 if i<thresh else 1 for i in all_predictions]

    tn, fp, fn, tp = confusion_matrix(all_targets, all_preds).ravel()
    print(tn, fp, fn, tp)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)


    print(f"Sensitivity: {sensitivity}")
    print(f"Specificity: {specificity}")
    print(f"AUC-ROC: {auc_roc}")

    # Plot ROC curve



    return test_loss, accuracy#, tn, fp, fn, tp,



def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            #print(output.shape)
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


def main():
    CHANNEL_WISE = 6
    random.seed(42)
    TEST = -1
    if TEST == 0:
        ff_layers = 1
        dset = EEGCwtDataset("ds003490-download", participants="participants.tsv",
             tstart=0, tend=240, batch_size=16, width=8, disk=True, epoched=False, prep=True,
             debug=False)

        # a = dset.split(ratios=(0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04),shuffle=True)
        a = dset.split(ratios=(0.3, 0.3,0.2), shuffle=True)
        #c = a[-1].split(ratios = (0.125,0.125,0.125,0.125,0.125,0.125,0.125,),balance_classes=False)

        for i in a:
                i.select_channel(-1)

        #[i.info() for i in c]
        #print("ATTENTION")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        num_classes = 2  # Replace with the number of classes you have
        model = ConvTimeAttentionV2(num_channels=512, num_classes=2, ff_layers=ff_layers).to(device)
        # model = CoherenceClassifier(num_classes, 63, 500).to(device)
        model = model.double()
        criterion = nn.CrossEntropyLoss()

        # Load the model from a saved checkpoint
        checkpoint_path = "/home/sebastjan/Documents/eeg/best_model_basic_TESTER_v2_att_cwt_split_0839_000001_001.pth"
        model.load_state_dict(torch.load(checkpoint_path))
        #for i in c:
        # Transfer test dataset to GPU
        a[-1].transfer_to_gpu(device)
        test_loader = torch.utils.data.DataLoader(a[-1], batch_size=16, shuffle=False)
        test_loss, test_accuracy = test2(model, device, test_loader, criterion)
        a[-1].delete_from_gpu(device)

        print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy * 100:.2f}%')
    if TEST == 1:
        ff_layers = 1
        dset = EEGCwtDataset("ds003490-download", participants="participants.tsv",
                             tstart=0, tend=240, batch_size=16, width=8, disk=True, epoched=False, prep=True,
                             debug=False)

        # a = dset.split(ratios=(0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04),shuffle=True)
        a = dset.split(ratios=(0.3, 0.3, 0.2), shuffle=True)
        #c = a[-1].split(ratios=(0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125,), balance_classes=False)
        c = a[-1].split(fractions=True, shuffle=False)
        a[-1].select_channel(-1)
        a[-1].info()
        #sleep(100)
        for i in c:
            i.select_channel(-1)

        [i.info() for i in c]
        # print("ATTENTION")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        num_classes = 2  # Replace with the number of classes you have
        model = ConvTimeAttentionV2(num_channels=512, num_classes=2, ff_layers=ff_layers).to(device)
        # model = CoherenceClassifier(num_classes, 63, 500).to(device)
        model = model.double()
        criterion = nn.CrossEntropyLoss()

        # Load the model from a saved checkpoint
        checkpoint_path = "/home/sebastjan/Documents/eeg/best_model_basic_TESTER_v2_att_cwt_split_0839_000001_001.pth"
        model.load_state_dict(torch.load(checkpoint_path))
        for i in c:
            i.transfer_to_gpu(device)
        test_loaders = [torch.utils.data.DataLoader(i, batch_size=16, shuffle=False) for i in c]
        test_loss, test_accuracy = test3(model, device, test_loaders, criterion)
        for i in c:
            i.delete_from_gpu(device)

        print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy * 100:.2f}%')

    if CHANNEL_WISE == 1:
        dset = EEGCwtDataset("ds003490-download", participants="participants.tsv",
                             tstart=0, tend=240, batch_size=256, debug=False, transform=add_dim, width=30)
        # Hyperparameters
        num_epochs = 10
        batch_size = 16
        learning_rate = 0.001

        dtrain, dtest = dset.split(ratios=0.8, shuffle=True, balance_classes=True)

        del dset
        for i in range(64):
            dtrain.select_channel(i)
            print("works?")
            dtest.select_channel(i)
            # dtrain.change_mode(ch=i)
            # dtest.change_mode(ch=i)
            train_loader = torch.utils.data.DataLoader(dtrain, batch_size=batch_size, shuffle=True)
            val_loader = torch.utils.data.DataLoader(dtest, batch_size=batch_size, shuffle=False)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            num_classes = 2  # Replace with the number of classes you have
            model = CoherenceClassifier(num_classes, 30, 500).to(device)
            model = model.double()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            criterion = nn.CrossEntropyLoss()

            best_val_accuracy = 0.0
            model_save_path = f"best_model{i}_cwt_one_ch.pth"

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
    elif CHANNEL_WISE == 2:
        # dset = EEGDataset("ds003490-download", participants="participants.tsv",
        #                  tstart=0, tend=240, batch_size=8, transform=reshaper,
        #                  trans_args=(transform_to_cwt,(np.linspace(1, 30, num=8), morlet2, True)))

        # dset = EEGDataset("ds003490-download", participants="participants.tsv",
        #                  tstart=0, tend=240, batch_size=64, name="_TESTER_clean",)
        #                  #transform=add_dim)  # transform=resizer, trans_args=(224,224))

        dset = EEGCwtDataset("ds003490-download", participants="participants.tsv",
                             tstart=0, tend=240, batch_size=64, width=8, disk=True, epoched=False, prep=True)

        # Hyperparameters
        num_epochs = 50
        batch_size = 256
        learning_rate = 0.0005

        # need to not transform
        dtrain, dtest = dset.split(0.8, shuffle=True)
        dtrain.select_channel(-1)
        dtest.select_channel(-1)

        # TODO PREDELI V PARTIAL TENSOR TO GPU!
        train_loader = torch.utils.data.DataLoader(dtrain, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = torch.utils.data.DataLoader(dtest, batch_size=batch_size, shuffle=False, num_workers=4)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        num_classes = 2  # Replace with the number of classes you have
        model = ConvTimeAttentionV2(num_channels=512, num_classes=2).to(device)
        # model = CoherenceClassifier(num_classes, 63, 500).to(device)
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
    elif CHANNEL_WISE == 3:
        # dset = EEGDataset("ds003490-download", participants="participants.tsv",
        #                  tstart=0, tend=240, batch_size=8, transform=reshaper,
        #                  trans_args=(transform_to_cwt,(np.linspace(1, 30, num=8), morlet2, True)))

        dset = EEGDataset("ds003490-download", participants="participants.tsv",
                          tstart=0, tend=240, batch_size=64, name="_TESTER_clean", disk=True, epoched=True)
        # transform=add_dim)  # transform=resizer, trans_args=(224,224))

        # dset = EEGCwtDataset("ds003490-download", participants="participants.tsv",
        #                     tstart=0, tend=240, batch_size=8, width=8, cache_len=3)
        #

        # Hyperparameters
        num_epochs = 50
        batch_size = 16
        learning_rate = 0.0005
        weight_decay = 0.001
        # need to not transform
        dtrain, dtest = dset.split(0.8, shuffle=True)
        # dtrain.select_channel(-1)
        # dtest.select_channel(-1)

        train_loader = torch.utils.data.DataLoader(dtrain, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(dtest, batch_size=batch_size, shuffle=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        num_classes = 2  # Replace with the number of classes you have
        model = ConvTimeAttentionV2(num_channels=63, num_classes=num_classes, ff_layers=2).to(device)
        # model = CoherenceClassifier(num_classes, 63, 500).to(device)
        model = model.double()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()

        best_val_accuracy = 0.0
        model_save_path = f"best_model_basic_TESTER_v2_att_filtered_2nd.pth"

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
    elif CHANNEL_WISE == 4:
        dset = EEGCwtDataset("ds003490-download", participants="participants.tsv",
                             tstart=0, tend=240, batch_size=16, width=8, disk=True, epoched=False, prep=True,
                             debug=False)
        # a = dset.split(ratios=(0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04),shuffle=True)
        a = dset.split(ratios=(0.3, 0.3,0.2), shuffle=True)
        for i in a:
            i.select_channel(-1)

        # a.delete_from_gpu('cuda')
        # Hyperparameters
        num_epochs = 200
        batch_size = 16
        learning_rate = 0.0001
        ff_layers = 1
        print(str(learning_rate))
        loss_values= []
        accuracy_values = []

        # train_loader =
        # val_loader = torch.utils.data.DataLoader(dtest, batch_size=batch_size, shuffle=False, num_workers=4)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        num_classes = 2  # Replace with the number of classes you have
        model = ConvTimeAttentionV2(num_channels=512, num_classes=2, ff_layers=ff_layers).to(device)
        # model = CoherenceClassifier(num_classes, 63, 500).to(device)
        model = model.double()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=[3,20,60],gamma=0.1)

        best_val_accuracy = 0.0
        model_save_path = f"best_model_basic_TESTER_v2_att_cwt_split.pth"
        rng = list(range(len(a) - 1))
        for epoch in range(1, num_epochs + 1):
            random.shuffle(rng)
            for i in rng:
                a[i].transfer_to_gpu(device)
                train_loader = torch.utils.data.DataLoader(a[i], batch_size=batch_size, shuffle=True)
                train(model, device, train_loader, optimizer, criterion, epoch)
                a[i].delete_from_gpu(device)

            # a[-1]
            a[-1].transfer_to_gpu(device)
            val_loader = torch.utils.data.DataLoader(a[-1], batch_size=batch_size, shuffle=False)
            val_loss, val_accuracy = test(model, device, val_loader, criterion)
            a[-1].delete_from_gpu(device)
            loss_values.append(val_loss)
            accuracy_values.append(val_accuracy)

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                model_save_path = f"best_model_cwt_3cnn_attention_{ff_layers}ff_"
                model_save_path = model_save_path + f"0{str(best_val_accuracy).split('.')[-1]}_"
                model_save_path = model_save_path + f"0{str(learning_rate).split('.')[-1]}_"
                model_save_path = model_save_path + f"epoch{epoch}.pth"
                torch.save(model.state_dict(), model_save_path)
                print(f"Model saved at {model_save_path}")
            checkpoint_interval = 5
            if epoch % checkpoint_interval == 0:
                save_checkpoint(model, optimizer, epoch, f'checkpoint_epoch_{epoch}.pth')
            scheduler.step()
    elif CHANNEL_WISE == 5:
        num_epochs = 50
        batch_size = 64
        learning_rate = 0.00001
        num_channels = [128,] #[128,64,32,16]#[128, 256,]# 512]
        ff_layers = 3
        num_classes = 2
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        dset = EEGDataset("ds003490-download", participants="participants.tsv",
                          tstart=0, tend=240, batch_size=16,) #name="_TESTER_clean",disk=True ,epoched=True)
        dtrain0, dtest = dset.split(ratios=0.8, shuffle=True)
        num_classes = 2
        train_loader0 = torch.utils.data.DataLoader(dtrain0, batch_size=batch_size, shuffle=True)
        #train_loader1 = torch.utils.data.DataLoader(dtrain1, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(dtest, batch_size=batch_size, shuffle=False)
        dtrain0.transfer_to_gpu(device)
        dtest.transfer_to_gpu(device)


        #print(len(dtrain0), len(dtrain1), len(dtest), len(dset))

        model = TCNClassifier(num_inputs=64, num_channels=num_channels, num_classes=num_classes, num_ff=ff_layers, kernel_size=5,dropout=0, batch_norm=False, attention=True).to(device)
        model = model.double()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)
        criterion = nn.CrossEntropyLoss()

        best_val_accuracy = 0.0
        model_save_path = f"best_model_basic_TESTER_v2_resnet1d.pth"

        for epoch in range(1, num_epochs + 1):
            #dtrain0.transfer_to_gpu(device)
            train(model, device, train_loader0, optimizer, criterion, epoch)
            #dtrain0.delete_from_gpu(device)
            #dtrain1.transfer_to_gpu(device)
            #train(model, device, train_loader1, optimizer, criterion, epoch)
            #dtrain1.delete_from_gpu(device)
            #dtest.transfer_to_gpu(device)
            val_loss, val_accuracy = test(model, device, val_loader, criterion)
            #dtest.delete_from_gpu(device)

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save(model.state_dict(),
                           f"best_model_basic_TESTER_v2_resnet1d_0{str(best_val_accuracy).split('.')[-1]}_0{str(learning_rate).split('.')[-1]}_{str(len(num_channels))}_{str(ff_layers)}.pth")
                print(f"Model saved at {model_save_path}")
            checkpoint_interval = 5
            if epoch % checkpoint_interval == 0:
                save_checkpoint(model, optimizer, epoch, f'checkpoint_epoch_{epoch}.pth')
    elif CHANNEL_WISE == 6:
        dset = EEGCwtDataset("ds003490-download", participants="participants.tsv",
                             tstart=0, tend=240, batch_size=16, width=8, disk=True, epoched=False, prep=True,
                             debug=False)
        checkpoint_path ="/home/sebastjan/Documents/eeg/best_model_cwt_2cnn_attention_2ff_0625836820083682_01e-06_epoch32_las_lay_seed-3236858436348559892.pth"
        num_epochs = 50
        batch_size = 16
        learning_rate = 0.000001
        ff_layers = 2
        torc_seed = torch.seed()
        print(str(learning_rate))

        a = dset.split(ratios=(0.3, 0.3,0.2), shuffle=True)
        for i in a:
            i.select_channel(-1)

        use_index = None
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        num_classes = 2  # Replace with the number of classes you have
        model = ConvTimeAttention(num_channels=512, num_classes=2).to(device)
        model.load_state_dict(torch.load(checkpoint_path))
        # model = CoherenceClassifier(num_classes, 63, 500).to(device)
        model = model.double()

        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()

        best_val_accuracy = 0.0
        model_save_path = f"best_model_basic_TESTER_v2_att_cwt_split_last_lay.pth"
        rng = list(range(len(a) - 1))
        random.shuffle(rng)
        for epoch in range(23, num_epochs + 1):
            for i in rng:
                a[i].transfer_to_gpu(device)
                train_loader = torch.utils.data.DataLoader(a[i], batch_size=batch_size, shuffle=True)
                train(model, device, train_loader, optimizer, criterion, epoch)
                a[i].delete_from_gpu(device)

            # a[-1]
            a[-1].transfer_to_gpu(device)
            val_loader = torch.utils.data.DataLoader(a[-1], batch_size=batch_size, shuffle=False)
            val_loss, val_accuracy = test2(model, device, val_loader, criterion)
            a[-1].delete_from_gpu(device)

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                model_save_path = f"best_model_cwt_2cnn_attention_{ff_layers}ff_"
                model_save_path = model_save_path + f"0{str(best_val_accuracy).split('.')[-1]}_"
                model_save_path = model_save_path + f"0{str(learning_rate).split('.')[-1]}_"
                model_save_path = model_save_path + f"epoch{epoch}_las_lay_seed-{torc_seed}.pth"
                torch.save(model.state_dict(), model_save_path)
                print(f"Model saved at {model_save_path}")
            checkpoint_interval = 5
            if epoch % checkpoint_interval == 0:
                pass
                #save_checkpoint(model, optimizer, epoch, f'checkpoint_epoch_{epoch}.pth')
    elif CHANNEL_WISE == 7:
        dset = EEGCwtDataset("ds003490-download", participants="participants.tsv",
                             tstart=0, tend=240, batch_size=16, width=8, disk=True, epoched=False, prep=True,
                             debug=False)
        #a = dset.split(ratios=(0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04),shuffle=True)
        a = dset.split(ratios=(0.3, 0.3,0.2), shuffle=True)
        for i in a:
            i.select_channel(-1)
        man_s = 42
        torch.manual_seed(man_s)
        # a.delete_from_gpu('cuda')
        # Hyperparameters
        num_epochs = 50
        batch_size = 16
        learning_rate = 0.000005
        ff_layers = 2
        print(str(learning_rate))
        loss_values= []
        accuracy_values = []

        # train_loader =
        # val_loader = torch.utils.data.DataLoader(dtest, batch_size=batch_size, shuffle=False, num_workers=4)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        num_classes = 2  # Replace with the number of classes you have
        model = ConvTimeAttentionReduction(num_channels=512, num_classes=2, ff_layers=ff_layers).to(device)
        # model = CoherenceClassifier(num_classes, 63, 500).to(device)
        model = model.double()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)
        w = torch.Tensor([0.6,0.4]).to(device)
        #criterion = nn.CrossEntropyLoss(weight=w)
        criterion = nn.CrossEntropyLoss()
        #scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=[3,20,60],gamma=0.1)

        best_val_accuracy = 0.0
        model_save_path = f"best_model_basic_TESTER_reduce_att_cwt_split.pth"
        rng = list(range(len(a) - 1))
        for epoch in range(1, num_epochs + 1):
            random.shuffle(rng)
            for i in rng:
                a[i].transfer_to_gpu(device)
                train_loader = torch.utils.data.DataLoader(a[i], batch_size=batch_size, shuffle=True)
                train(model, device, train_loader, optimizer, criterion, epoch)
                a[i].delete_from_gpu(device)

            # a[-1]
            a[-1].transfer_to_gpu(device)
            val_loader = torch.utils.data.DataLoader(a[-1], batch_size=batch_size, shuffle=False)
            val_loss, val_accuracy = test2(model, device, val_loader, criterion)
            a[-1].delete_from_gpu(device)
            loss_values.append(val_loss)
            accuracy_values.append(val_accuracy)

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                model_save_path = f"best_model_cwt_3cnn_reduce_attention_{ff_layers}ff_"
                model_save_path = model_save_path + f"0{str(best_val_accuracy).split('.')[-1]}_"
                model_save_path = model_save_path + f"0{str(learning_rate).split('.')[-1]}_"
                model_save_path = model_save_path + f"epoch{epoch}_mans{man_s}.pth"
                torch.save(model.state_dict(), model_save_path)
                print(f"Model saved at {model_save_path}")
            checkpoint_interval = 5
            if epoch % checkpoint_interval == 0:
                save_checkpoint(model, optimizer, epoch, f'checkpoint_epoch_{epoch}.pth')
            #scheduler.step()
    elif CHANNEL_WISE == 8:
        dset = EEGCwtDataset("ds003490-download", participants="participants.tsv",
                             tstart=0, tend=240, batch_size=16, width=8, disk=True, epoched=False, prep=True,
                             debug=False)
        #a = dset.split(ratios=(0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04),shuffle=True)
        a = dset.split(ratios=(0.3, 0.3,0.2), shuffle=True)
        for i in a:
            i.select_channel((1,2,7,29))
        man_s = 42
        torch.manual_seed(man_s)
        # a.delete_from_gpu('cuda')
        # Hyperparameters
        num_epochs = 200
        batch_size = 16
        learning_rate = 0.000001
        ff_layers = 1
        print(str(learning_rate))
        loss_values= []
        accuracy_values = []

        # train_loader =
        # val_loader = torch.utils.data.DataLoader(dtest, batch_size=batch_size, shuffle=False, num_workers=4)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        num_classes = 2  # Replace with the number of classes you have
        model = ConvTimeAttentionV2(num_channels=512, num_classes=2, ff_layers=ff_layers).to(device)
        # model = CoherenceClassifier(num_classes, 63, 500).to(device)
        model = model.double()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)
        #criterion = nn.CrossEntropyLoss(weight=w)
        criterion = nn.CrossEntropyLoss()
        #scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=[3,20,60],gamma=0.1)

        best_val_accuracy = 0.0
        model_save_path = f"best_model_basic_TESTER_reduce_att_cwt_split.pth"
        rng = list(range(len(a) - 1))
        for epoch in range(1, num_epochs + 1):
            random.shuffle(rng)
            for i in rng:
                a[i].transfer_to_gpu(device)
                train_loader = torch.utils.data.DataLoader(a[i], batch_size=batch_size, shuffle=True)
                train(model, device, train_loader, optimizer, criterion, epoch)
                a[i].delete_from_gpu(device)

            # a[-1]
            a[-1].transfer_to_gpu(device)
            val_loader = torch.utils.data.DataLoader(a[-1], batch_size=batch_size, shuffle=False)
            val_loss, val_accuracy = test2(model, device, val_loader, criterion)
            a[-1].delete_from_gpu(device)
            loss_values.append(val_loss)
            accuracy_values.append(val_accuracy)

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                model_save_path = f"best_model_cwt_3cnn_reduce_attention_{ff_layers}ff_"
                model_save_path = model_save_path + f"0{str(best_val_accuracy).split('.')[-1]}_"
                model_save_path = model_save_path + f"0{str(learning_rate).split('.')[-1]}_"
                model_save_path = model_save_path + f"epoch{epoch}_mans{man_s}.pth"
                torch.save(model.state_dict(), model_save_path)
                print(f"Model saved at {model_save_path}")
            checkpoint_interval = 5
            if epoch % checkpoint_interval == 0:
                save_checkpoint(model, optimizer, epoch, f'checkpoint_epoch_{epoch}.pth')
            #scheduler.step()
    plt.figure()
    plt.plot(range(1,num_epochs+1), loss_values)
    plt.title('Loss over time')
    plt.xlabel('Epcohs')
    plt.ylabel('Loss')
    plt.show()

if __name__ == "__main__":
    main()
