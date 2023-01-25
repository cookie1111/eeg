import numpy as np
from scipy.signal import cwt, morlet2
import matplotlib.pyplot as plt
from mne.io import read_raw_eeglab, read_raw_brainvision
from torch.utils.data import DataLoader, Dataset
from typing import Literal
from pathlib import Path

import os
import mne
import torch
from torch.utils.data import Dataset, DataLoader

class EEGDataset(Dataset):
    def __init__(self, root_dir, tmin=0, tmax=30):
        self.root_dir = root_dir
        self.tmin = tmin
        self.tmax = tmax
        self.subjects = os.listdir(self.root_dir)
        self.epochs_list = []
        self.load_data()

    def __len__(self):
        return len(self.epochs_list)

    def __getitem__(self, idx):
        return self.epochs_list[idx]

    def load_data(self):
        for subject in self.subjects:
            subject_path = os.path.join(self.root_dir, subject)
            eeg_file = os.path.join(subject_path, [f for f in os.listdir(subject_path) if f.endswith('.set')][0])
            raw = mne.io.read_raw_eeglab(eeg_file, preload=True)
            # mark the rest state part
            rest_events = mne.make_fixed_length_events(raw, id=1, start=0, stop=raw.times[-1], duration=30)
            raw.add_events(rest_events)
            # mark the oddball part
            oddball_events = mne.make_fixed_length_events(raw, id=2, start=30, stop=raw.times[-1], duration=30)
            raw.add_events(oddball_events)
            # extract only the rest state part
            rest_epochs = mne.Epochs(raw, rest_events, tmin=self.tmin, tmax=self.tmax, preload=True, baseline=None)
            self.epochs_list.append(rest_epochs)


# this is for one singular record
class Singular_EEG_Dataset(Dataset):

    def __init__(self, file_path:str, file_type:str = "eeg_lab"):
        """
        Init the location and the type of the signal and read it

        :param file_path: path to the file
        :param file_type: eeg_lab, raw_brainvision
        """

        if file_type == "eeg_lab":
            self.sig = read_raw_eeglab(file_path)
        elif file_type == "raw_brainvision":
            self.sig = read_raw_brainvision(file_path)
        else:
            raise ValueError("Wrong type of file")
        self.data_framed = self.sig.to_data_frame()

    def window(self, win_size):
        pass

    def wavelet_transform_single_channel(self):
        pass

    def wavelet_transform_all_channels(self):
        pass


class Participant_Singular(Dataset):
    def __init__(self, path, mode: Literal["rest", "patterns"] = "rest",
                 classification: Literal["parkinsons",  "medication"] = "parkinsons"):
        pass

    def get_EEG_datsets(self):
        pass

# TODO: change to DataPipe
# multiple participants dataset
class Participants_Dataset(Dataset):

    def __init__(self, root_folder):
        self.root = Path(root_folder)

    def __len__(self):
        return sum(1 if "sub" in str(p) else 0 for p in self.root.iterdir())

    def __getitem__(self, item: int) -> Participant_Singular:
        """
        The folders are named with index starting at 1 but the indexing in dataset is done from zero.
        Participants_Dataset[0] returns sub-001
        :param item: works only with indexing
        :return: returns Participant_Singular of the selected subject
        """
        cnt = 0
        for p in self.root.iterdir():
            if "sub" in str(p):
                if cnt == item:
                    return Participant_Singular(p)
                cnt = cnt + 1
        raise IndexError(f"{item} is out of range for {len(self)}")
