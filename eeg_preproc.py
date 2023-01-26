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

# first session is without medication
# annotations are already added on the thing first 4mins (til s201 marker is rest state)
# S 3 and S 4 are eyes closed, S 1 and S 2 are eyes oepened higher orders are auditory signals
# if person has 2 ses folders it means its a patient else its a control

class EEGDataset(Dataset):
    def __init__(self, root_dir, tstart=0, tend=30, special_part=None, medicated=0):
        """
        Grab all subjects, for now only the medicated session is supported, add a class field to the whole thing and
        window their eeg signal slice(slice is based off special_part parameter). Windows are accessed via index, and
        one index is one window. Datasplit is performed on the subject level, meaning when we split for 80% we will be
        excluding subjects and not a part of a subject.

        :param root_dir: root directory of the dataset
        :param tstart: start of the slice to use for windowing operation, measured in seconds
        :param tend: end of slice to use for windowing operation, measured in seconds resting state as a whole lasts
        till 4 mins
        :param special_part: if we have a part based off annotations that we wish to use(annotations are hardcoded into
        the class S 1, S 2, S 3, S 4 for closed eyes and open eyes respectively)
        :param medicated: 0 - use only the medicated data, 1 - use only the off-medication data, 2 - use both
        medicated and off-medication data.
        """
        self.root_dir = root_dir
        self.tstart = tstart
        self.tend = tend
        self.special_part = special_part
        self.medicated = medicated
        self.subjects = [p for p in os.listdir(self.root_dir) if "sub-" in p and os.path.isdir(p)]
        self.epochs_list = None
        self.y_list = []
        self.load_data()

    def __len__(self):
        return len(self.y_list)

    def __getitem__(self, idx):
        return (self.epochs_list[idx,:,:],self.y_list[idx])

    def load_data(self):
        for subject in self.subjects:
            subject_path = os.path.join(self.root_dir, subject)
            # subject class, if its 0 the subject has PD if 1 its a control
            y = 1 if len(os.listdir(subject_path)) == 1 else 0
            if self.medicated == 0 or len(os.listdir(subject_path)) == 1:
                subject_path = os.path.join(subject_path, "ses-01")
            elif self.medicated == 1:
                subject_path = os.path.join(subject_path, "ses-02")
            else:
                subject_path1 = os.path.join(subject_path, "ses-01")
                subject_path2 = os.path.join(subject_path, "ses-02")
            if not self.medicated == 2:
                eeg_file = os.path.join(subject_path, [f for f in os.listdir(subject_path) if f.endswith('.set')][0])
                raw = mne.io.read_raw_eeglab(eeg_file, preload=True)

                if not self.special_part:
                    rest_events = mne.make_fixed_length_events(raw, id=1, start=self.tstart, stop=self.tend,
                                                               duration=2, overlap=1.9)
                    rest_epochs = mne.Epochs(raw, rest_events, 1, self.tstart, self.tend).get_data()
                    accompanying_y = [y]*rest_epochs.shape[0]

            else:
                eeg_file1 = os.path.join(subject_path1, [f for f in os.listdir(subject_path1) if f.endswith('.set')][0])
                raw1 = mne.io.read_raw_eeglab(eeg_file1, preload=True)
                eeg_file2 = os.path.join(subject_path2, [f for f in os.listdir(subject_path2) if f.endswith('.set')][0])
                raw2 = mne.io.read_raw_eeglab(eeg_file2, preload=True)

            # mark the rest state part
            #rest_events = mne.make_fixed_length_events(raw, id=1, start=0, stop=raw.times[-1], duration=30)
            #raw.add_events(rest_events)
            # mark the oddball part
            #oddball_events = mne.make_fixed_length_events(raw, id=2, start=30, stop=raw.times[-1], duration=30)
            #raw.add_events(oddball_events)
            # extract only the rest state part
            # = mne.Epochs(raw, rest_events, tstart=self.tstart, tend=self.tend, preload=True, baseline=None)
            #TODO fix so its not a python list but a numpy array!!!
            if self.epochs_list is None:
                self.epochs_list = rest_epochs
            else:
                np.concatenate([self.epochs_list.append, rest_epochs], axis=0)

            self.y_list.append(accompanying_y)


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
