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
mne.set_log_level("DEBUG")
import torch
from torch.utils.data import Dataset, DataLoader


# first session is without medication
# annotations are already added on the thing first 4mins (til s201 marker is rest state)
# S 3 and S 4 are eyes closed, S 1 and S 2 are eyes oepened higher orders are auditory signals
# if person has 2 ses folders it means its a patient else its a control
class EEGDataset(Dataset):
    def __init__(self, root_dir: str, participants: str, id_column: str = "participant_id", tstart: int = 0,
                 tend: int = 30, special_part: str = None, medicated: int = 0, cache_amount: int = 1,
                 batch_size: int = 16):
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
        :param batch_size: amount of datapoints to be preloaded is used for switching in and loading the next epochs from
        file since they are too large to hold all in RAM
        """
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.participants = participants
        self.tstart = tstart
        self.tend = tend
        self.special_part = special_part
        self.medicated = medicated
        self.ids = id_column
        self.subjects = pd.read_table(os.path.join(root_dir, participants))
        self.epochs_list = []
        self.y_list = []
        # holds first 2 epochs loads a new one when we are at the end of the first one should be queue
        # cur holds end of epoch index and the epoch itself
        self.cache = []
        self.cache_pos = 0
        self.data_points = []
        self.cache_size = cache_amount
        self.load_data()
        self.semafor = False

    def __len__(self):
        return sum(self.data_points)

    # change so it doesn't load the whole thing and cache the current subject for fast access

    def __getitem__(self, idx: int):
        idx_epoch, idx_inner = self.convert_to_idx(idx)
        idx_next_epoch, idx_inner = self.convert_to_idx(idx+self.batch_size)
        if idx_epoch + 1 == idx_next_epoch and not self.semafor:
            self.semafor = True
            self.load_next_epoch(idx_next_epoch)
            self.semafor = False
        # random access VERY INEFFICIENT
        elif idx_epoch + 1 < idx_next_epoch:
            self.cache_pos = idx_epoch
            self.load_particular_epoch(idx_epoch)
        return self.cache[idx_epoch-self.cache_pos][1][idx_inner].get_data(), self.y_list[idx_epoch]
        # return self.epochs_list[idx_epoch].get_data()[idx_inner], self.y_list[idx_epoch]

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
        #check if column exists for name of file
        fresh_entries = True
        f_name = f"lens_{self.medicated}_{self.tstart}_{self.tend}_noDrop_epo"
        if f_name in self.subjects:
            fresh_entries =False
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
                # print(subject_path)
                subject_path_eeg = os.path.join(subject_path,os.listdir(subject_path)[0])
                # print(subject_path_eeg)
                eeg_file = os.path.join(subject_path_eeg,
                                        [f for f in os.listdir(subject_path_eeg) if f.endswith('.set')][0])

                if not self.special_part:
                    # print(self.tstart,self.tend)
                    save_dest = os.path.join(subject_path_eeg, f"{self.medicated}_{self.tstart}_{self.tend}_noDrop_epo.fif")
                    if os.path.isfile(save_dest):
                        if fresh_entries:
                            rest_epochs = mne.read_epochs(save_dest)
                    else:
                        raw = mne.io.read_raw_eeglab(eeg_file, preload=True)
                        low_cut = 0.1
                        hi_cut = 30
                        raw = raw.filter(low_cut, hi_cut)
                        raw = raw.crop(tmin=self.tstart, tmax=self.tend)
                        raw = raw.drop_channels(["X", "Y", "Z"])
                        rest_epochs = mne.make_fixed_length_epochs(raw,duration=2, overlap=1.9).drop_bad()
                        rest_epochs.save(save_dest)
                    if len(self.cache) < self.cache_size:

                        """rest_epochs = mne.Epochs(raw,
                                                 rest_events,
                                                 1,
                                                 self.tstart,
                                                 self.tend,
                                                 baseline=(None, None)).drop_bad()"""
                        print(f"Caching {len(self.cache)+1}/{self.cache_size}")
                        # have index saved for easier cache checking
                        rest_epochs = mne.read_epochs(save_dest)
                        self.cache.append((len(rest_epochs) + (0 if len(self.cache) == 0 else self.cache[-1][0]),
                                           rest_epochs))

                    #mne.save(os.path.join(subject_path,'saved_epoch.fif', overwrite=True), rest_epochs)
            else:
                eeg_file1 = os.path.join(subject_path1, [f for f in os.listdir(subject_path1) if f.endswith('.set')][0])
                raw1 = mne.io.read_raw_eeglab(eeg_file1)
                eeg_file2 = os.path.join(subject_path2, [f for f in os.listdir(subject_path2) if f.endswith('.set')][0])
                raw2 = mne.io.read_raw_eeglab(eeg_file2)

            if self.epochs_list is None:
                # print("ye")
                #need to save filenames here
                self.epochs_list = [save_dest]
                if fresh_entries:
                    self.data_points = [len(rest_epochs)]
                else:
                    self.data_points.append(eval('subject.'+f_name))
            else:
                # print("nay")
                self.epochs_list.append(save_dest)
                if fresh_entries:
                    self.data_points.append(len(rest_epochs))
                else:
                    self.data_points.append(eval('subject.'+f_name))
            self.y_list.append(y)
        if fresh_entries:
            self.subjects[f_name] = self.data_points
            self.subjects.to_csv(os.path.join(self.root_dir, self.participants), sep="\t", index=False, na_rep="nan")

    def load_next_epoch(self, idx_next_epoch):
        print(f"loading epoch {idx_next_epoch} from {self.epochs_list[idx_next_epoch]}")
        #if the queue is larger than len(self.epochs_list) then remove the first one
        if len(self.epochs_list) > self.cache_size:
            self.cache.pop()
            self.cache_pos = self.cache_pos + 1
        epoch = mne.read_epochs(self.epochs_list[idx_next_epoch])
        self.cache.append(((len(epoch) + self.cache[-1][0]),
                           epoch))

    #this is extremely inefficient!!!! dataset shouldn't be random accessed!
    def load_particular_epoch(self, idx_epoch):
        l = sum(self.data_points[:idx_epoch])
        epoch = mne.read_epochs(self.epochs_list[idx_epoch])
        self.cache = [l, epoch]

if __name__ == '__main__':
    dset = EEGDataset("./ds003490-download", participants="participants.tsv",
                      tstart=0, tend=240, cache_amount=2, batch_size=8)
    for i in range(10000):
        dset[i]
        print(i)
    """
    dloader = DataLoader(dset, batch_size=8, shuffle=False, num_workers=1)
    cnt = 0
    for i in dloader:
        print(cnt)
        cnt = cnt+1
    """