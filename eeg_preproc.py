import numpy as np
import sys
from scipy.signal import cwt, morlet2
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt
from mne.io import read_raw_eeglab, read_raw_brainvision
from torch.utils.data import DataLoader, Dataset
from typing import Literal, Tuple
from pathlib import Path
import pandas as pd
import cv2
import os
import mne
mne.set_log_level("DEBUG")
import torch
from torch.utils.data import Dataset, DataLoader
from math import ceil
import random
from wavelets import calculate_cwt_coherence
"""
coherence has to be implemented seperatly since we need to do node mixing so i should construct it as a seperate process

NEED TO FIGURE OUT WHICH NODES AND HOW MANY WIDTHS TO USE -> how do i build the transform function to be used in the dataset also i should fix preloading

"""

#def optical_flow(matrix


def resizer(matrix, new_x, new_y):
    # might have to assert float type
    #print(f"Matrix is of dtype: {matrix.dtype}")
    print(f"shape: {matrix.shape}, new_shapes: {new_x}.{new_y}")
    matrix = np.squeeze(matrix)
    matrix = cv2.resize(matrix,(new_x,new_y))
    return np.repeat(matrix[np.newaxis,:,:],3,axis=0)

def transform_to_cwt(signals, widths, wavelet):
    new_signals = []
    for i in signals.shape[0]:
        new_signals.append(cwt(signals[i,:,:], wavelet, widths))

    return new_signals

class EEGNpDataset(Dataset):

    def __init__(self, root_dir: str, participants: str, id_column: str = "participant_id", tstart: int = 0,
                 tend: int = 30, special_part: str = None, medicated: int = 0,
                 batch_size: int = 16, use_index = None, duration: float = 1, overlap: float = 0.9,
                 stack_rgb = True, transform = lambda x:x, trans_args = (), freq = 500):
        self.freq = freq
        self.overlap = overlap
        self.duration = duration
        self.stack_em = stack_rgb
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.participants = participants
        self.tstart = tstart
        self.tend = tend
        self.special_part = special_part
        self.medicated = medicated
        self.ids = id_column
        self.subjects = pd.read_table(os.path.join(root_dir, participants))
        # for using a custom index
        if use_index is not None:
            self.subjects = self.subjects.iloc[use_index]
        self.egg_list = []
        self.y_list = []
        self.data_points = []
        self.epochs_list = []
        self.load_data()
        self.stack = stack_rgb
        self.transform = transform
        self.trans_args = trans_args

    def load_data(self):
        fresh_entries = True
        f_name = f"len_{self.medicated}_{self.tstart}_{self.tend}_noDrop_{self.overlap}_{self.duration}_np"
        f_name = f_name.replace('.','d')
        if f_name in self.subjects:
            fresh_entries=False

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
                    save_dest = os.path.join(subject_path_eeg, f"{self.medicated}_{self.tstart}_{self.tend}_noDrop_{self.overlap}_{self.duration}_np")
                    save_dest = save_dest.replace('.','d')+'.fif'
                    if os.path.isfile(save_dest):
                        arr = np.load(save_dest)
                    else:
                        raw = mne.io.read_raw_eeglab(eeg_file, preload=True)
                        low_cut = 0.1
                        hi_cut = 30
                        raw = raw.filter(low_cut, hi_cut)
                        raw = raw.crop(tmin=self.tstart, tmax=self.tend)
                        raw = raw.drop_channels(["X", "Y", "Z"])
                        eeg_nfo = raw.info
                        print(eeg_nfo.get("hpi_meas"))
                        arr = raw.get_data()
                        np.save(save_dest, arr)
            else:
                eeg_file1 = os.path.join(subject_path1, [f for f in os.listdir(subject_path1) if f.endswith('.set')][0])
                raw1 = mne.io.read_raw_eeglab(eeg_file1)
                eeg_file2 = os.path.join(subject_path2, [f for f in os.listdir(subject_path2) if f.endswith('.set')][0])
                raw2 = mne.io.read_raw_eeglab(eeg_file2)

            if self.epochs_list is None:
                self.epochs_list = [arr]
                l = self.calc_samples(arr)
                self.data_points = [l]
            else:
                self.epochs_list.append(arr)
                l = self.calc_samples(arr)
                self.data_points.append(l)
            self.y_list.append(y)
        # we save the dataframe
        if fresh_entries:
            self.subjects[f_name] = self.data_points
            subjects = self.subjects.sort_index()
            subjects.to_csv(os.path.join(self.root_dir, self.participants), sep="\t", index=False, na_rep="nan")
        print(self.data_points)

    def calc_samples(self, arr):
        whole = arr.shape[1]
        duration = self.duration*self.freq
        overlap = self.overlap*self.freq
        return int(np.floor((whole-duration)/(duration-overlap)))

    def __getitem__(self, idx: int):
        print(idx)
        duration = self.duration * self.freq
        try:
            idx_subject, idx_inner = self.convert_to_idx(idx)
        except TypeError:
            print(idx,duration)
            print("error encountered something in convert to idx went wrong and the function didn't reach return condition and returned None")
            sys.exit(1)
        print(f"schmove: {idx_inner}:{idx_inner+duration}, {idx_subject}")
        # print(len(self.epochs_list), self.epochs_list[0].shape)
        return self.transform(self.epochs_list[idx_subject][:, idx_inner:(idx_inner+duration)],
                              *self.trans_args), self.y_list[idx_subject]

    def __len__(self):
        suma = sum(self.data_points)
        return suma

    def convert_to_idx(self, index):
        suma = 0
        idx = 0
        if index >= len(self):
            raise IndexError
        duration = self.duration * self.freq
        overlap = self.overlap * self.freq
        step = duration - overlap
        for i in self.data_points:
            if suma + i>index:
                cur = index-suma
                return int(idx), int(cur*step)
            idx = idx + 1
            suma = suma + i


# first session is without medication
# annotations are already added on the thing first 4mins (til s201 marker is rest state)
# S 3 and S 4 are eyes closed, S 1 and S 2 are eyes oepened higher orders are auditory signals
# if person has 2 ses folders it means its a patient else its a control
class EEGDataset(Dataset):
    def __init__(self, root_dir: str, participants: str, id_column: str = "participant_id", tstart: int = 0,
                 tend: int = 30, special_part: str = None, medicated: int = 0, cache_amount: int = 1,
                 batch_size: int = 16, use_index = None, duration: float = 1, overlap: float = 0.9, stack_rgb = True, transform = lambda x:x, trans_args = ()):
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
        :param shuffle: wether to shuffle the participants when reading the dataset
        """
        self.overlap = overlap
        self.duration = duration
        self.stack_em = stack_rgb
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.participants = participants
        self.tstart = tstart
        self.tend = tend
        self.special_part = special_part
        self.medicated = medicated
        self.ids = id_column
        self.subjects = pd.read_table(os.path.join(root_dir, participants))
        if use_index is not None:
            print(use_index)
            self.subjects = self.subjects.iloc[use_index]
        print(self.subjects.head)
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
        self.stack = stack_rgb
        self.transform = transform
        self.trans_args = trans_args
        self.whole = False


    def split(self, ratios = 0.8, shuffle: bool = False):
        """
        splits the dataset into 2 datasets, make sure you set the caching of the higher order dset to what the split
        datasets will use, keep in mind your memory size! both are held in ram and initialized!! Ratios are used in the
        sense of previous ratio is the starting index and next ratio is the ending index

        :param ratios: splits in percentiles, the amount of ratios you input will result in that many splits
        :param shuffle: wether to shuffle the participants befor splitting (not necessary if the original dataset is already shuffeled)
        :return: len(ratios)+1 datasets containing indexes based on ratios
        """
        shuffled_idxes =  list(range(len(self.y_list)))
        if shuffle:
            random.shuffle(shuffled_idxes)
        if ratios is None:
            return self

        elif isinstance(ratios, float) or len(ratios) == 1:
            idx = ceil(len(self.y_list)*ratios)
            return (EEGDataset(self.root_dir, self.participants, self.ids, self.tstart, self.tend, self.special_part,
                self.medicated,self.cache_size, self.batch_size, use_index=shuffled_idxes[:idx], transform=self.transform, trans_args=self.trans_args, overlap=self.overlap, duration=self.duration),
                    EEGDataset(self.root_dir, self.participants, self.ids, self.tstart, self.tend, self.special_part,
                               self.medicated, self.cache_size, self.batch_size, use_index=shuffled_idxes[idx:len(self.y_list)],transform=self.transform, trans_args=self.trans_args,overlap=self.overlap, duration=self.duration))
        else:
            assert isinstance(ratios, tuple)
            splits = []
            prev_idx = 0
            for ratio in ratios:
                idx = ceil(len(self.y_list) * ratio)
                splits.append(
                    EEGDataset(self.root_dir, self.participants, self.ids, self.tstart, self.tend, self.special_part,
                    self.medicated, self.cache_size, self.batch_size, use_index=shuffled_idxes[prev_idx: idx]))
                prev_idx = idx
            splits.append(
                EEGDataset(self.root_dir, self.participants, self.ids, self.tstart, self.tend, self.special_part,
                           self.medicated, self.cache_size, self.batch_size,
                           use_index=shuffled_idxes[prev_idx: len(self.y_list)]))
            return splits

    def preload_whole(self):
        self.whole = True
        self.cache = []
        for epoch in self.epochs_list:
            self.cache.append(mne.read_epochs(epoch))

    def __len__(self):
        return sum(self.data_points)

    def __getitem__(self, idx: int):
        idx_epoch, idx_inner = self.convert_to_idx(idx)
        if self.whole:
            return self.transform(self.cache[idx_epoch][idx_inner].get_data(), *self.trans_args), self.y_list[idx_epoch]
        if idx_epoch + 1 < len(self.epochs_list):
            idx_next_epoch, idx_next_inner = self.convert_to_idx(idx+self.batch_size)
        else:
            idx_next_epoch = idx_epoch
            idx_next_inner = idx_inner
        if idx_epoch + 1 == idx_next_epoch and not self.semafor:
            self.semafor = True
            self.load_next_epoch(idx_epoch+self.cache_size)
            self.semafor = False
        # random access VERY INEFFICIENT
        elif idx_epoch + 1 < idx_next_epoch:
            self.cache_pos = idx_epoch
            self.load_particular_epoch(idx_epoch)
        # print(self.cache)
        ret =  self.transform(self.cache[idx_epoch-self.cache[0][0]][1][idx_inner].get_data(), *self.trans_args), self.y_list[idx_epoch]
        # return self.epochs_list[idx_epoch].get_data()[idx_inner], self.y_list[idx_epoch]
        #print(ret[0].shape)
        return ret

    def convert_to_idx(self, index):
        suma = 0
        idx = 0
        for i in self.data_points:
            suma = suma + i
            if index < suma:
                return idx, index+i-suma

            idx = idx + 1

    def load_data(self):
        cnt = 0
        #check if column exists for name of file
        fresh_entries = True
        f_name = f"len_{self.medicated}_{self.tstart}_{self.tend}_noDrop_{self.overlap}_{self.duration}_epo"
        f_name = f_name.replace('.','d')
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
                    save_dest = os.path.join(subject_path_eeg,
                                             f"{self.medicated}_{self.tstart}_{self.tend}_noDrop_{self.overlap}_{self.duration}_epo")
                    #save_dest = os.path.join(subject_path_eeg, f"{self.medicated}_{self.tstart}_{self.tend}_noDrop_epo.fif")
                    save_dest = save_dest.replace('.','#')+'.fif'
                    if os.path.isfile(save_dest):
                        os.remove(save_dest)
                    save_dest = os.path.join(subject_path_eeg, f"{self.medicated}_{self.tstart}_{self.tend}_noDrop_{self.overlap}_{self.duration}_epo")
                    save_dest = save_dest.replace('.','d')+'.fif'
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
                        rest_epochs = mne.make_fixed_length_epochs(raw,duration=self.duration, overlap=self.overlap).drop_bad()
                        rest_epochs.save(save_dest)
                    if len(self.cache) < self.cache_size:

                        """rest_epochs = mne.Epochs(raw,
                                                 rest_events,
                                                 1,
                                                 self.tstart,
                                                 self.tend,
                                                 baseline=(None, None)).drop_bad()"""
                        # print(f"Caching {len(self.cache)+1}/{self.cache_size}")
                        # have index saved for easier cache checking
                        rest_epochs = mne.read_epochs(save_dest)
                        self.cache.append((len(self.epochs_list), rest_epochs))

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
            subjects = self.subjects.sort_index()
            subjects.to_csv(os.path.join(self.root_dir, self.participants), sep="\t", index=False, na_rep="nan")

    def load_next_epoch(self, epoch_to_load):
        if self.cache[-1][0] == epoch_to_load:
            # epoch is already loaded
            print(f"epoch {epoch_to_load} is already cached")
            return None
        if self.cache[-1][0] +1 == len(self.epochs_list):
            print(f"Last epoch already cached")
            return None
        print(f"loading epoch {epoch_to_load} from {self.epochs_list[self.cache[-1][0]+1]}, highest epoch is {self.cache[-1][0]}")
        #if the queue is larger than len(self.epochs_list) then remove the first one
        if len(self.epochs_list) > self.cache_size:
            self.cache.pop(0)
        epoch = mne.read_epochs(self.epochs_list[epoch_to_load])
        self.cache.append((epoch_to_load, epoch))

    #this is extremely inefficient!!!! dataset shouldn't be random accessed!
    def load_particular_epoch(self, idx_epoch):
        l = sum(self.data_points[:idx_epoch])
        epoch = mne.read_epochs(self.epochs_list[idx_epoch])
        self.cache = [l, epoch]

    def clear_cache(self):
        self.cache = []

TEST = 1
if __name__ == '__main__':
    if TEST == 0:
        dset = EEGDataset("ds003490-download", participants="participants.tsv",
                          tstart=0, tend=240, cache_amount=1, batch_size=8,)#transform=resizer, trans_args=(224,224))
        #need to not transform
        dset_train, dset_train1, dset_test = dset.split((0.4,0.8), shuffle = True)
        print(len(dset_train),len(dset_train1), len(dset_test))
        print(dset_train.subjects, dset_test.subjects)
        del dset
        dset_train1.clear_cache()
        dset_test.clear_cache()


        dset_train.preload_whole()

        sys.exit(0)
        dloader = DataLoader(dset, batch_size=8, shuffle=False, num_workers=1)

        for step, (x,y) in enumerate(dloader):
            print(x.shape, y.shape)
            lino = x[0,0,0,:]
            print(lino.shape, "shapin")

            long_boi = np.real(cwt(lino,morlet2,np.logspace(np.log2(2),np.log2(50),num=23)))
            short_boi = np.real(cwt(lino,morlet2,np.logspace(np.log2(2),np.log2(50),num=8)))
            print(long_boi.shape)
            long_boi = resizer(long_boi,500,500)
            short_boi = resizer(short_boi,500,500)


            fig, axs = plt.subplots(2)
            axs[0].imshow(short_boi[0,:,:])
            axs[0].set_title('short_boi')
            axs[1].imshow(long_boi[0,:,:])
            axs[1].set_title('_boi')
            plt.show()
            break

            print(step)

    elif TEST == 1:
        ds = EEGNpDataset("ds003490-download", participants="participants.tsv",
                          tstart=0, tend=240, batch_size=8,transform=resizer,trans_args=(224,224))
        dl = DataLoader(ds, batch_size=32,num_workers=4,shuffle=True)
        t = 3
        cnt = 0
        for step, i in enumerate(dl):
            print(i)
            """cnt = cnt + 1
            if cnt == t:
                break"""

    """NEED TO DECIDE WETHER I WANNA SPLIT THE DATASET BY PARTICIPANTS OR BY SIGNALS"""