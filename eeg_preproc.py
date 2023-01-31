import numpy as np
from scipy.signal import cwt, morlet2
import matplotlib.pyplot as plt
from mne.io import read_raw_eeglab, read_raw_brainvision
from torch.utils.data import DataLoader, Dataset
from typing import Literal
from pathlib import Path
import pandas as pd

import os
import mne
import torch
from torch.utils.data import Dataset, DataLoader


# first session is without medication
# annotations are already added on the thing first 4mins (til s201 marker is rest state)
# S 3 and S 4 are eyes closed, S 1 and S 2 are eyes oepened higher orders are auditory signals
# if person has 2 ses folders it means its a patient else its a control
class EEGDataset(Dataset):
    def __init__(self, root_dir: str, participants: str, id_column: str = "participant_id", tstart: int = 0,
                 tend: int = 30, special_part: str = None, medicated: int = 0, cache_amount: int = 1):
        """
        Grab all subjects, for now only the medicated session is supported, add a class field to the whole thing and
        window their eeg signal slice(slice is based off special_part parameter). Windows are accessed via index, and
        one index is one window. Datasplit is performed on the subject level, meaning when we split for 80% we will be
        excluding subjects and not a part of a subject.

        :param root_dir: root directory of the dataset
        :param participants: file describing the participants, only provide the file name and make sure its in the
        root dir
        :param id_column: name of column which includes the participant ids in the participants file
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
        self.ids = id_column
        self.subjects = pd.read_table(os.path.join(root_dir, participants))
        self.epochs_list = None
        self.y_list = []
        # holds first 2 epochs loads a new one when we are at the end of the first one should be queue
        # cur holds end of epoch index and the epoch itself
        self.cache = []
        self.data_points = []
        self.cache_size = cache_amount
        self.load_data()

    def __len__(self):
        return sum(self.data_points)

    # change so it doesn't load the whole thing and cache the current subject for fast access
    def __getitem__(self, idx: int):
        idx_epoch, idx_inner = self.convert_to_idx(idx)
        return self.epochs_list[idx_epoch].get_data()[idx_inner], self.y_list[idx_epoch]

    def convert_to_idx(self, index):
        suma = 0
        idx = 0
        for i in self.data_points:
            suma = suma + i
            if index < suma:
                return idx, index+i-suma

            idx = idx + 1

    # TODO add option to save epochs to disk and load them
    def load_data(self):
        cached = 0
        self.cache = []
        for subject in self.subjects.itertuples():
            subject_path = os.path.join(self.root_dir, subject.participant_id)
            # subject class, if its 0 the subject has PD if 1 its a control
            y = 1 if subject.Group == "CTL" else 0
            if self.medicated == 0 or y == 1:
                if subject.sess1_Med == "OFF":
                    subject_path = os.path.join(subject_path, "ses-02")
                else:
                    subject_path = os.path.join(subject_path, "ses-01")
            elif self.medicated == 1:
                if subject.sess1_Med == "OFF":
                    subject_path = os.path.join(subject_path, "ses-01")
                else:
                    subject_path = os.path.join(subject_path, "ses-02")
            else:
                if y == 1:
                    continue
                subject_path1 = os.path.join(subject_path, "ses-01")
                subject_path2 = os.path.join(subject_path, "ses-02")
            if not self.medicated == 2:
                print(subject_path)
                subject_path_eeg = os.path.join(subject_path,os.listdir(subject_path)[0])
                print(subject_path_eeg)
                eeg_file = os.path.join(subject_path_eeg,
                                        [f for f in os.listdir(subject_path_eeg) if f.endswith('.set')][0])
                raw = mne.io.read_raw_eeglab(eeg_file, preload=True)
                raw.crop(tmin=self.tstart, tmax=self.tend)

                if not self.special_part:
                    rest_events = mne.make_fixed_length_events(raw, id=1, duration=2, overlap=1.9)
                    print(self.tstart,self.tend)
                    save_dest = os.path.join(subject_path_eeg, f"{self.medicated}_{self.tstart}_{self.tend}_epo.fif")
                    if len(self.cache) < self.cache_size:
                        if os.path.isfile(save_dest):
                            rest_epochs = mne.read_epochs(save_dest)
                        else:
                            rest_epochs = mne.Epochs(raw,
                                                     rest_events,
                                                     1,
                                                     self.tstart,
                                                     self.tend,
                                                     baseline=None).drop_bad()

                            rest_epochs.save(save_dest)
                        # have index saved for easier cache checking
                        self.cache.append((len(rest_epochs)+0 if len(self.cache) == 0 else self.cache[-1][0],
                                           rest_epochs))

                        self.epochs_list.append(save_dest)
                    else:
                        rest_epochs = mne.Epochs(raw,
                                                 rest_events,
                                                 1,
                                                 self.tstart,
                                                 self.tend,
                                                 baseline=None).drop_bad()
                    #mne.save(os.path.join(subject_path,'saved_epoch.fif', overwrite=True), rest_epochs)
            else:
                eeg_file1 = os.path.join(subject_path1, [f for f in os.listdir(subject_path1) if f.endswith('.set')][0])
                raw1 = mne.io.read_raw_eeglab(eeg_file1)
                eeg_file2 = os.path.join(subject_path2, [f for f in os.listdir(subject_path2) if f.endswith('.set')][0])
                raw2 = mne.io.read_raw_eeglab(eeg_file2)

            if self.epochs_list is None:
                print("ye")
                #need to save filenames here
                self.epochs_list = [rest_epochs]
                self.data_points = [len(rest_epochs)]
            else:
                print("nay")
                self.epochs_list.append(rest_epochs)
                self.data_points.append(len(rest_epochs))

            self.y_list.append(y)


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


if __name__ == '__main__':
    dset = EEGDataset("/home/sebastjan/PycharmProjects/eeg/ds003490-download", participants="participants.tsv",
                      tstart=0, tend=10)