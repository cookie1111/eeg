import numpy as np
from scipy.signal import cwt, morlet2
import matplotlib.pyplot as plt
from mne.io import read_raw_eeglab, read_raw_brainvision
from torch.utils.data import DataLoader, Dataset
from typing import Literal


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

# multiple participants dataset
class Participants_Dataset(Dataset):

    def __init__(self, root_folder):
        root_folder
