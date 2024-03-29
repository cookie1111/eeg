import numpy as np
import sys
from scipy.signal import cwt, morlet2
import sklearn
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
from math import ceil, prod
import random
from wavelets import calculate_cwt_coherence
import seaborn as sns
from scipy.signal import coherence
from sklearn.decomposition import PCA

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def node_wise_classification(dtrain, dtest, node, accuracy, precision, recall, f1):
    train_loader = DataLoader(dtrain,32,shuffle=True)
    test_loader = DataLoader(dtest,32,shuffle=True)
    clf = SGDClassifier(loss='hinge', alpha=1 / (1 * len(train_loader)), max_iter=1000, tol=1e-3, random_state=42)

    for batch in train_loader:
        features, labels = batch
        features = features.view(features.size(0), -1)
        clf.partial_fit(features.numpy(),labels.numpy(),np.unique(labels.numpy()))
    dtrain.clear_cache()
    y_test, y_pred = [],[]
    for batch in test_loader:
        features, labels = batch
        features = features.view(features.size(0), -1)
        y_test.extend(labels.numpy())
        y_pred.extend(clf.predict(features.numpy()))
    dtest.clear_cache()
    # Calculate performance metrics
    accuracy[node] = accuracy_score(y_test, y_pred)
    precision[node] = precision_score(y_test, y_pred, average='weighted')
    recall[node] = recall_score(y_test, y_pred, average='weighted')
    f1[node] = f1_score(y_test, y_pred, average='weighted')

    return accuracy, precision, recall, f1


def eeg_to_rgb_coherence_matrix(eeg_data, fs, nperseg):
    num_channels = eeg_data.shape[0]

    # Compute the coherence matrix
    coherence_matrix = np.zeros((num_channels, num_channels))
    for i in range(num_channels):
        for j in range(num_channels):
            if i == j:
                coherence_matrix[i, j] = 1
            elif i < j:
                f, c = coherence(eeg_data[i], eeg_data[j], fs, nperseg=nperseg)
                coherence_matrix[i, j] = np.mean(c)
            else:
                coherence_matrix[i, j] = coherence_matrix[j, i]
        # Normalize the coherence matrix to [0, 1]
        coherence_matrix = (coherence_matrix - np.min(coherence_matrix)) / (
                    np.max(coherence_matrix) - np.min(coherence_matrix))

        # Convert the coherence matrix to an RGB image (using PCA)
    pca = PCA(n_components=3)
    reshaped_coherence_matrix = np.reshape(coherence_matrix, (-1, num_channels))
    transformed_coherence_matrix = pca.fit_transform(reshaped_coherence_matrix)
    rgb_coherence_matrix = np.reshape(transformed_coherence_matrix, (num_channels, num_channels, 3))

    # Resize the image to 224x224
    resized_image = cv2.resize(rgb_coherence_matrix, (224, 224), interpolation=cv2.INTER_LINEAR)

    return resized_image


def coherence_between_matrices(matrix1, matrix2):
    # Check that the matrices have the same shape
    assert matrix1.shape == matrix2.shape, "Matrices have different shapes"
    n_electrodes, n_samples = matrix1.shape

    # Calculate the cross-spectral density matrix between the two matrices
    cxy = np.zeros((n_electrodes, n_samples), dtype=np.complex128)
    for i in range(n_electrodes):
        cxy[i] = np.multiply(np.fft.fft(matrix1[i]), np.fft.fft(matrix2[i]).conj())

    # Calculate the power spectral density of each matrix
    pxx = np.zeros((n_electrodes, n_samples), dtype=np.complex128)
    for i in range(n_electrodes):
        pxx[i] = np.multiply(np.fft.fft(matrix1[i]), np.fft.fft(matrix1[i]).conj())
    pyy = np.zeros((n_electrodes, n_samples), dtype=np.complex128)
    for i in range(n_electrodes):
        pyy[i] = np.multiply(np.fft.fft(matrix2[i]), np.fft.fft(matrix2[i]).conj())

    # Calculate the magnitude-squared coherence (MSC) between the rows of the matrices
    msc = np.zeros(n_electrodes)
    for i in range(n_electrodes):
        msc[i] = np.abs(cxy[i]).mean() ** 2 / (pxx[i].mean() * pyy[i].mean())

    return msc


def load_subject_data(root_folder):
    """
    Loads numpy arrays from subfolders of the given root folder.

    Args:
    - root_folder (str): Path to the root folder containing subject subfolders.

    Returns:
    - data (dict): A dictionary containing the loaded numpy arrays, with subject IDs as keys.
    """
    # Initialize empty dictionary to hold data
    data = []

    # Loop over subject subfolders
    for i in range(1, 51):
        # Format subject number with leading zeros
        sub_id = f"{i:03}"

        # Create path to subject subfolder
        sub_folder = os.path.join(root_folder, f"sub-{sub_id}")

        # Check if subfolder exists and contains 2 subfolders
        if os.path.exists(sub_folder) and len(os.listdir(sub_folder)) == 2:
            # Loop over subfolders in the subject subfolder
            compare = []
            for subdir in os.listdir(sub_folder):
                subdir_path = os.path.join(sub_folder, subdir)
                if os.path.isdir(subdir_path):
                    npy_subfolder = os.path.join(subdir_path, "eeg")
                    if os.path.exists(npy_subfolder):
                        # Loop over possible filenames
                        for j in [0, 1]:
                            npy_path = os.path.join(npy_subfolder, f"{j}_0_240_noDrop_0d9_1_np.npy")
                            if os.path.exists(npy_path):
                                compare.append(np.load(npy_path))
            data.append(coherence_between_matrices(compare[0], compare[1]))
    return data


def delete_file_with_name(root_folder_path="ds003490-download"):
    for dirpath, dirnames, filenames in os.walk(root_folder_path):
        for filename in filenames:
            if filename == "0_0_240_noDrop_0.9_1_epo.fif":
                file_path = os.path.join(dirpath, filename)
                os.remove(file_path)
                print(f"Deleted {file_path}")


def resizer(matrix, new_x, new_y, transform=None, transform_args=None, add_dims = True):
    # might have to assert float type
    if transform:
        matrix = transform(matrix,*transform_args)
    matrix = np.squeeze(matrix)
    matrix = cv2.resize(matrix,(new_x,new_y))
    if add_dims:
        return np.repeat(matrix[np.newaxis,:,:],3,axis=0)
    else:
        if matrix.shape[-1] == 3:
            matrix = np.transpose(2, 0, 1)
        return matrix

"""
need to figure out how to transform the cwt data to look presentable/recognisable for the model
"""

def get_balanced_with_depth_divisable_by_3(matrix):
    sorts = np.argsort(matrix.shape)

    sh = matrix.shape
    d = sh[sorts[0]]/3
    assert d % 1 == 0
    small = sh[sorts[1]]*int(d)

    big = sh[sorts[2]]
    if big < small:
        i_big = small
        small = big
        big = i_big
    if big == small:
        return small,big,3

    besto = 0
    #it doesn't matter which one of the first two dimensions is the smaller we will reshape them anyway
    change = 1
    while big-change > small+change:
        if (big-change)*(small+change) == big*small:
            besto = change

        change = change + 1
    return small+besto,big-besto,3

def reshaper(signals, transform=None, transform_args=None):
    if transform:
        signals = transform(signals, *transform_args)

    x,y,b = get_balanced_with_depth_divisable_by_3(signals)
    resa = np.reshape(signals,(x,y,b),'F')
    return resa

def transform_to_cwt(signals, widths, wavelet, real=True,transform=None, transform_args=None):

    if transform:
        signals = transform(signals,*transform_args)

    new_signals = []
    #print(signals.shape)
    for i in range(signals.shape[0]):
        if real:
            new_signals.append(np.real(cwt(signals[i,:], wavelet, widths)))
        else:
            new_signals.append(np.imag(cwt(signals[i,:], wavelet, widths)))

    #print(np.array(new_signals).shape)
    return np.array(new_signals)

class EEGNpDataset(Dataset):

    def __init__(self, root_dir: str, participants: str, id_column: str = "participant_id", tstart: int = 0,
                 tend: int = 30, special_part: str = None, medicated: int = 1,
                 batch_size: int = 16, use_index = None, duration: float = 1, overlap: float = 0.9, name: str = "",
                 stack_rgb = True, transform = lambda x:x, trans_args = (), freq = 500, debug = False, disk = False, epoched = False):
        self.name = name
        self.disk = disk
        self.freq = freq
        self.debug = debug
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
        self.epoched = epoched
        self.use_index = use_index
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
        self.ch = -1


    def transfer_to_gpu(self, device):
        """
        Copy the cwt_cache to the specified device.
        """
        self.holder = [torch.tensor(epoch).to(device) for epoch in self.epochs_list]
        self.epochs_list, self.holder = self.holder, self.epochs_list


    def delete_from_gpu(self, device):
        self.epochs_list = self.holder
        self.holder = []
        torch.cuda.empty_cache()

    def change_mode(self, ch=-1):
        # ch being -1 means return all in get_item
        if self.debug:
            print(f"DEBUG:Changing channel to {ch} from {self.ch}")
        self.ch = ch

    def load_data(self):
        fresh_entries = True
        f_name = f"len_{self.medicated}_{self.tstart}_{self.tend}_noDrop_{self.overlap}_{self.duration}_np{self.name}"
        f_name = f_name.replace('.','d')
        if self.debug:
            print(f"DEBUG: Checking if {f_name} already mentioned in the subjects ds.")
        if f_name in self.subjects:
            fresh_entries=False
        if self.debug:
            print(f"DEBUG: {f_name} is {'not' if fresh_entries else ''} in subjects")

        for subject in self.subjects.itertuples():

            subject_path = os.path.join(self.root_dir, subject.participant_id)
            # subject class, if its 0 the subject has PD if 1 its a control
            y = 1 if subject.Group == "CTL" else 0
            if self.debug:
                print(f"DEBUG: subject is in {'CTL' if y else 'PD'} group")

            # if self.medicated == 0 means that the subject was medicated -> looking for session where subject was medicated
            # if self.medicated == 1 means that the subject wasn't medicated
            # if self.medicated == 2 means that we use both the medicated and not medicated portion for training
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
            if self.debug:
                print(f"DEBUG: subject was {'OFF' if self.medicated == 1 else 'ON'} medication")
            if not self.medicated == 2:
                if self.debug:
                    print(f"DEBUG: using only one session per subject")

                # print(subject_path)
                subject_path_eeg = os.path.join(subject_path, os.listdir(subject_path)[0])
                if self.debug:
                    print(f"DEBUG: loading subjects eeg from {subject_path_eeg}")
                # print(subject_path_eeg)
                eeg_file = os.path.join(subject_path_eeg,
                                        [f for f in os.listdir(subject_path_eeg) if f.endswith('.set')][0])
                if not self.special_part:
                    save_dest = os.path.join(subject_path_eeg, f"{self.medicated}_{self.tstart}_{self.tend}_noDrop_{self.overlap}_{self.duration}_np{self.name}")
                    save_dest = save_dest.replace('.','d')+'npy.npy'
                    if os.path.isfile(save_dest):
                        os.remove(save_dest)
                    save_dest = os.path.join(subject_path_eeg, f"{self.medicated}_{self.tstart}_{self.tend}_noDrop_{self.overlap}_{self.duration}_np{self.name}")
                    save_dest = save_dest.replace('.', 'd')+".npy"

                    if self.debug:
                        print(f"DEBUG: preparing save_destination for subject {save_dest}")

                    if os.path.isfile(save_dest):
                        if self.debug:
                            print(f"DEBUG: {save_dest} already exists so no need to load it")
                        arr = np.load(save_dest, mmap_mode='r' if self.disk else None)
                    else:
                        if self.debug:
                            print(f"DEBUG: {save_dest} does not yet exist so filtering and cutting from scratch")
                        raw = mne.io.read_raw_eeglab(eeg_file, preload=True)
                        low_cut = 1
                        hi_cut = 30
                        raw = raw.filter(low_cut, hi_cut)
                        raw = raw.crop(tmin=self.tstart, tmax=self.tend)
                        raw = raw.drop_channels(["X", "Y", "Z"])
                        #eeg_nfo = raw.info
                        #print(eeg_nfo.get("hpi_meas"))
                        arr = raw.get_data()
                        np.save(save_dest, arr)
                        if self.debug:
                            print(f"DEBUG: Preprocessed version saved to {save_dest}")
            else:
                eeg_file1 = os.path.join(subject_path1, [f for f in os.listdir(subject_path1) if f.endswith('.set')][0])
                raw1 = mne.io.read_raw_eeglab(eeg_file1)
                eeg_file2 = os.path.join(subject_path2, [f for f in os.listdir(subject_path2) if f.endswith('.set')][0])
                raw2 = mne.io.read_raw_eeglab(eeg_file2)

            if self.epochs_list is None:
                if self.debug:
                    print(f"DEBUG: epochs_list has yet to be populated adding arr to epochs_list")

                self.epochs_list = [arr]
                l = self.calc_samples(arr)
                if self.debug:
                    print(f"DEBUG: arr info = {arr.shape}, number of samples= {l}")
                self.data_points = [l]
            else:
                if self.debug:
                    print(f"DEBUG: epochs_list adding arr to epochs_list")
                self.epochs_list.append(arr)
                l = self.calc_samples(arr)
                if self.debug:
                    print(f"DEBUG: arr info = {arr.shape}, number of samples= {l}")
                self.data_points.append(l)

            self.y_list.append(y)
            if self.debug:
                print(f"DEBUG: {subject} has the class {y} meaning its {'PD' if y == 0 else 'CTL'}")
        # we save the dataframe
        if fresh_entries:
            if self.debug:
                print(f"DEBUG: Adding the subject to the df since its a fresh entry")
            self.subjects[f_name] = self.data_points
            subjects = self.subjects.sort_index()
            subjects.to_csv(os.path.join(self.root_dir, self.participants), sep="\t", index=False, na_rep="nan")
        print(self.subjects)

    def calc_samples(self, arr):
        if self.epoched:
            return arr.shape[0]
        else:
            whole = arr.shape[1]
            duration = self.duration * self.freq
            overlap = self.overlap * self.freq
            ret = int(np.floor((whole - duration) / (duration - overlap)))
            if self.debug:
                print(f"DEBUG: calculating number of samples in the recording")
                print(f"duration_of_sample = {duration}")
                print(f"overlap between consecutive samples = {overlap}")
                print(f"number of samples = {ret}")
            return ret

    def __getitem__(self, idx: int):
        if self.debug:
            print(f"DEBUG: Fetching sample number {idx}")
        duration = self.duration * self.freq
        try:
            if self.debug:
                print(f"DEBUG: converting {idx} (overall index) to subject_number (denoting the subject to \n"
                      f"be loaded not equivalent to subject id) and within sample_index (index of sample within a subj\n"
                      f"ect)")
            idx_subject, idx_inner = self.convert_to_idx(idx)
            if self.debug:
                print(f"DEBUG: subject_number = {idx_subject}\n"
                      f"       sample_index = {idx_inner}")
        except TypeError:
            print(idx,duration)
            print("error encountered something in convert to idx went wrong and the function didn't reach return condition and returned None")
            sys.exit(1)
        if self.ch == -1:
            if self.debug:
                print(f"DEBUG: using all channels and running transformation on top of them, with {'preepoched' if self.epcohed else 'nonepoched'} data")
            if self.epoched:
                #print(self.epochs_list[idx_subject].shape)
                return self.transform(self.epochs_list[idx_subject][idx_inner,:,:],
                                      *self.trans_args), self.y_list[idx_subject]
            else:
                return self.transform(self.epochs_list[idx_subject][:, idx_inner:(idx_inner + duration)],
                                      *self.trans_args), self.y_list[idx_subject]
        else:
            if self.debug:
                print(f"DEBUG: running on channel {self.ch} and applying transformation, with {'preepoched' if self.epcohed else 'nonepoched'} data")
            if self.epoched:
                return self.transform(np.expand_dims(self.epochs_list[idx_subject][idx_inner, self.ch, :], axis=0),
                                      *self.trans_args), self.y_list[idx_subject]
            else:
                return self.transform(
                    np.expand_dims(self.epochs_list[idx_subject][self.ch, idx_inner:(idx_inner + duration)], axis=0),
                    *self.trans_args), self.y_list[idx_subject]

    def __len__(self):
        suma = sum(self.data_points)
        return suma

    def convert_to_idx(self, index):
        suma = 0
        idx = 0
        step = 1
        if index >= len(self):
            raise IndexError
        if not self.epoched:
            duration = self.duration * self.freq
            overlap = self.overlap * self.freq
            step = duration - overlap

        for i in self.data_points:
            if suma + i>index:
                cur = index-suma
                return int(idx), int(cur*step)
            idx = idx + 1
            suma = suma + i

    def split(self,ratios=0.8, shuffle=False, balance_classes=True):
        if balance_classes:
            #need to know how much one and the other list have
            c1 = [idx for idx,c in enumerate(self.y_list) if c == 1]
            c0 = [idx for idx,c in enumerate(self.y_list) if c == 0]
            if shuffle:
                random.shuffle(c1)
                random.shuffle(c0)
        else:
            shuffled_idxes = list(range(len(self.y_list)))
            if shuffle:
                random.shuffle(shuffled_idxes)

        if ratios is None:
            return self
        elif isinstance(ratios, float) or len(ratios) == 1:
            if balance_classes:
                ce1 = ceil(len(c1)*ratios)
                ce0 = ceil(len(c0)*ratios)
                bottom = c1[:ce1]+c0[:ce0]
                top = c1[ce1:] + c0[ce0:]
            else:
                idx = ceil(len(self.y_list)*ratios)
                bottom = shuffled_idxes[:idx]
                top = shuffled_idxes[idx:]

            return (EEGNpDataset(self.root_dir, self.participants, self.ids, self.tstart, self.tend, self.special_part,
                                 self.medicated, self.batch_size, use_index=bottom, transform=self.transform, trans_args=self.trans_args,
                                 overlap=self.overlap, duration=self.duration, debug=False, disk=self.disk, epoched=self.epoched, name=self.name),
                    EEGNpDataset(self.root_dir, self.participants, self.ids, self.tstart, self.tend, self.special_part,
                                 self.medicated, self.batch_size, use_index=top,transform=self.transform, trans_args=self.trans_args,
                                 overlap=self.overlap, duration=self.duration, debug=False, disk=self.disk, epoched=self.epoched, name=self.name))
        else:
            assert isinstance(ratios, tuple)
            splits = []
            if balance_classes:
                prev_idx1 = 0
                prev_idx0 = 0
            else:
                prev_idx = 0
            for ratio in ratios:
                if balance_classes:
                    ce1 = ceil(len(c1)*ratio)
                    ce0 = ceil(len(c0)*ratio)
                    bottom = c1[prev_idx1:prev_idx1+ce1]+c0[prev_idx0:prev_idx0+ce0]
                else:
                    idx = ceil(len(self.y_list) * ratio)
                    bottom = shuffled_idxes[prev_idx: iprev_idx + idx]
                splits.append(
                    EEGNpDataset(self.root_dir, self.participants, self.ids, self.tstart, self.tend, self.special_part,
                    self.medicated, self.batch_size, use_index=bottom, disk=self.disk, epoched=self.epoched, name=self.name))
                if balance_classes:
                    prev_idx1 = prev_idx1 + ce1
                    prev_idx0 = prev_idx0 + ce0
                else:
                    prev_idx = prev_idx + idx

            if balance_classes:
                bottom = c1[prev_idx1:]+c0[prev_idx0:]
            else:
                bottom = shuffled_idxes[prev_idx:]

            splits.append(
                EEGNpDataset(self.root_dir, self.participants, self.ids, self.tstart, self.tend, self.special_part,
                           self.medicated, self.batch_size,
                           use_index=bottom, disk=self.disk, epoched=self.epoched, name=self.name))
            return splits

    def info(self):
        self.debug = False
        print(f"________________________INFO________________________\n"
              f"Amount of subjects: {len(self.epochs_list)}\n"
              f"Participants: {self.subjects.participant_id}"
              f"Amount of subjects based of data_points: {len(self.data_points)}\n"
              f"All samples: {sum(self.data_points)}\n"
              f"Class balance: {sum(self.y_list)/len(self.y_list)} (1 class vs all)\n"
              f"Get random variable: {self[random.randint(0,len(self))]}\n")
        self.debug = False

# first session is without medication
# annotations are already added on the thing first 4mins (til s201 marker is rest state)
# S 3 and S 4 are eyes closed, S 1 and S 2 are eyes oepened higher orders are auditory signals
# if person has 2 ses folders it means its a patient else its a control
class EEGDataset(Dataset):
    def __init__(self, root_dir: str, participants: str, id_column: str = "participant_id", tstart: int = 0,
                 tend: int = 30, special_part: str = None, medicated: int = 0, cache_amount: int = 1,
                 batch_size: int = 16, use_index = None, duration: float = 1, overlap: float = 0.9,
                 stack_rgb = True, transform = lambda x:x, trans_args = ()):
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
        self.name = name
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


if __name__ == '__main__':
    TEST = 6
    if TEST == 0:
        dset = EEGNpDataset("ds003490-download", participants="participants.tsv",
                          tstart=0, tend=240, batch_size=8,)#transform=resizer, trans_args=(224,224))
        #need to not transform
        dset_train, dset_test = dset.split(0.8, shuffle = True)
        print(len(dset_train), len(dset_test))
        print(dset_train.subjects, dset_test.subjects)

        dloader = DataLoader(dset, batch_size=8, shuffle=False, num_workers=1)

        for step, (x,y) in enumerate(dloader):
            print(x.shape, y.shape)
            lino = x[0,0,:]
            print(lino.shape, "shapin")

            #TODO use linspace over log space or maybe have to change it and invert the logspace so that it focuses on the lower? frequencies!

            #long_boi = np.real(cwt(lino,morlet2,np.logspace(np.log2(2),np.log2(50),num=23)))
            long_boi = np.real(cwt(lino,morlet2,np.linspace(1,30,num=23)))
            #short_boi = np.real(cwt(lino,morlet2,np.logspace(np.log2(2),np.log2(50),num=8)))
            short_boi = np.real(cwt(lino,morlet2,np.linspace(1,30,num=8)))
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
                          tstart=0, tend=240, batch_size=8,transform=resizer,trans_args=(224,224,transform_to_cwt,(np.linspace(1,30,num=23),morlet2,True)))
        dl = DataLoader(ds, batch_size=32,num_workers=1,shuffle=True)
        ds.change_mode(ch=1)
        #ds.split(ratios=0.8,shuffle=True)
        t = 3
        cnt = 0
        for step, i in enumerate(dl):
            print(i[0].shape)
        """cnt = cnt + 1
        if cnt == t:
            break"""
        ds.change_mode(ch=2)
        dl = DataLoader(ds, batch_size=32,num_workers=1,shuffle=True)
        for step, i in enumerate(dl):
            print(i[0].shape)
            break
    elif TEST == 2:
        ds = EEGNpDataset("ds003490-download", participants="participants.tsv",
                          tstart=0, tend=240, batch_size=8,transform=resizer,trans_args=(224,224,transform_to_cwt,(np.linspace(1,30,num=23),morlet2,True)))

        ds.split(0.8, shuffle=True, balance_classes=True)
    elif TEST == 3:
        ds = EEGNpDataset("ds003490-download", participants="participants.tsv",
                          tstart=0, tend=240, batch_size=8,
                          transform=reshaper,
                          trans_args=(224, 224, 3,
                                      transform_to_cwt, (np.linspace(1, 30, num=23), morlet2, True)),
                          debug = True)
        dtrain,dtest = ds.split(0.8, shuffle=True, balance_classes=True)
        del ds
        dl = DataLoader(dtrain,batch_size=4,shuffle=True,num_workers=1)
        dlt = DataLoader(dtest,batch_size=4,shuffle=True,num_workers=1)
        for i in dl:
            print("w")
    elif TEST == 4:
        dset = EEGNpDataset("ds003490-download", participants="participants.tsv",
                            tstart=0, tend=240, batch_size=8, medicated=1,) #transform=resizer, trans_args=(224,224))
        # need to not transform
        dset.info()
        print(dset[0])
        res = transform_to_cwt(dset[0][0],np.linspace(1,30,num=24),morlet2,True)
        print(res.shape)
        print(get_balanced_with_depth_divisable_by_3(res))
        res_re = (reshaper(res))
        print(res_re.shape)
        res_res = resizer(res_re, 224,224, add_dims=False)
        print(res_res.shape)
        print(np.max(res), np.max(res_re))
        fig, axs = plt.subplots(4)
        axs[0].imshow(sklearn.preprocessing.minmax_scale(res[0,:,:]))
        res_sca = sklearn.preprocessing.minmax_scale(res_re)
        axs[1].imshow(res_sca[:50,:,:])
        axs[2].imshow(res_sca[50:100,:,:])
        axs[3].imshow(res_sca[100:150,:,:])
        plt.show()
    elif TEST == 5:
        EEGNpDataset("ds003490-download", participants="participants.tsv",
                     tstart=0, tend=240, batch_size=8, medicated=0, )
        #21*64 but with 2second interval(1000 steps) -> resize with cubic interpolation
        # search for which signal contributes the most difference between on medication and off medication in patients
        #delete_file_with_name()
        pass
    elif TEST == 6:
        data = np.array(load_subject_data("ds003490-download"))

        # Set electrode and subject names
        subject_names = [f"Sub {i + 1}" for i in range(data.shape[0])]
        electrode_names = [f"El {i + 1:03d}" for i in range(data.shape[1])]

        averages = np.mean(data, axis=0)
        sorted_indices = np.argsort(averages)[::-1]

        # Create a new array with the top 5 columns set to 1 and the rest set to 0
        mask = np.zeros(data.shape[1])
        mask[sorted_indices[-5:]] = 1

        # Create heatmap using seaborn and apply the mask to highlight the columns
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(data, cmap="coolwarm", xticklabels=electrode_names, yticklabels=subject_names, ax=ax,
                    mask=np.broadcast_to(mask, data.shape))

        # Set plot title and axis labels
        ax.set_xlabel("Subjects")
        ax.set_ylabel("Electrodes")
        ax.set_title("Top 5 Columns by Average Value")

        # Show plot
        plt.show()
    elif TEST == 7:
        pass
