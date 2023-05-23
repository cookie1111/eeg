import numpy as np
from scipy.signal import cwt, morlet2
import os
import mne
from eeg_preproc import EEGNpDataset, node_wise_classification
import os
import unittest
import numpy as np
import random
from math import ceil
import torch
import json
import pandas as pd
from autoreject import AutoReject

DEBUG = False



def debug_print( message):
    if DEBUG:
        print(f"DEBUG: {message}")

class EEGCwtDataset(EEGNpDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cwt_cache = [None] * len(self.epochs_list)
        self.parts = len(self.epochs_list)
        self.widths = np.linspace(1, 30, num=8)
        # self.debug = True

    def clear_cache(self):
        self.cwt_cache = [None] * self.parts

    def apply_cwt(self, data: np.ndarray, channel: int):
        # print(data[channel].shape,)
        if channel == -1:
            # Apply CWT to all channels and stack the results
            cwt_results = [cwt(data[ch], morlet2, self.widths) for ch in range(data.shape[0])]
            cwt_result = np.concatenate(cwt_results, axis=1)
        else:
            cwt_result = cwt(data[channel], morlet2, self.widths)
        return np.array(cwt_result)

    def select_channel(self, channel: int):
        self.clear_cache()
        #self.reset()
        #gc.collect()
        self.load_data()
        if channel == -1:
            self.cwt_cache = [None] * len(self.epochs_list)
            self.ch = channel
            index = 0
            while self.epochs_list:
                epoch = self.epochs_list.pop(0)  # Remove and return the first element
                self.cwt_cache[index] = np.real(self.apply_cwt(epoch, channel))
                index += 1
        elif 0 <= channel < self.epochs_list[0].shape[0]:
            self.cwt_cache = [None] * len(self.epochs_list)
            self.ch = channel
            index = 0
            while self.epochs_list:
                epoch = self.epochs_list.pop(0)  # Remove and return the first element
                self.cwt_cache[index] = np.real(self.apply_cwt(epoch, channel))
                index += 1
        else:
            raise ValueError(f"Invalid channel. Channel must be between 0 and {self.epochs_list[0].shape[0] - 1}.")

    def __getitem__(self, idx: int):
        if self.debug:
            print(f"DEBUG: Fetching sample number {idx}")
        duration = self.duration * self.freq
        try:
            if self.debug:
                print(f"DEBUG: converting {idx} (overall index) to subject_number (denoting the subject to \n"
                      f"be loaded not equivalent to subject id) and within sample_index (index of sample within a subj\n"
                      f"ect)")
            # print(idx, len(self))
            idx_subject, idx_inner = self.convert_to_idx(idx)
            if self.debug:
                print(f"DEBUG: subject_number = {idx_subject}\n"
                      f"       sample_index = {idx_inner}")
        except TypeError:
            print(idx, duration)
            print(
                "error encountered something in convert to idx went wrong and the function didn't reach return condition and returned None")
            sys.exit(1)

        if self.ch == -1:
            raise ValueError("A channel must be selected before getting items.")
        else:
            if self.debug:
                print(f"DEBUG: running on channel {self.ch} and applying CWT from cache")
                print(f"{self.cwt_cache[idx_subject].shape} cwt shape")
            # print(idx_subject)
            cwt_data = self.cwt_cache[idx_subject][:, idx_inner:(idx_inner + duration)]
            return self.transform(cwt_data, *self.trans_args), self.y_list[idx_subject]

    def load_data(self):
        subjects = pd.read_table(os.path.join(self.root_dir, self.participants))
        f_name = f"len_{self.medicated}_{self.tstart}_{self.tend}_noDrop_{self.overlap}_{self.duration}_np_TESTER_clean".replace(
            '.', 'd')
        fresh_entries = f_name not in subjects
        debug_print(f"{f_name} is {'not' if fresh_entries else ''} in subjects")
        self.y_list = []
        self.epochs_list = None
        self.data_points = None
        montage = mne.channels.make_standard_montage('standard_1020')

        for subject in subjects.itertuples():
            subject_path = os.path.join(self.root_dir, subject.participant_id)
            y = 1 if subject.Group == "CTL" else 0
            debug_print(f"subject is in {'CTL' if y else 'PD'} group")

            if self.medicated in [0, 2] or y == 1:
                session = "ses-02" if subject.sess1_Med == "OFF" else "ses-01"
            else:
                session = "ses-01" if subject.sess1_Med == "OFF" else "ses-02"
            subject_path = os.path.join(subject_path, session)

            debug_print(f"subject was {'OFF' if self.medicated == 1 else 'ON'} medication")
            debug_print(f"using only one session per subject")

            subject_path_eeg = os.path.join(subject_path, os.listdir(subject_path)[0])
            debug_print(f"loading subjects eeg from {subject_path_eeg}")

            eeg_file = os.path.join(subject_path_eeg,
                                    next(f for f in os.listdir(subject_path_eeg) if f.endswith('.set')))
            save_dest = os.path.join(subject_path_eeg,
                                     f"{self.medicated}_{self.tstart}_{self.tend}_noDrop_{self.overlap}_{self.duration}_np_TESTER_clean".replace(
                                         '.', 'd') + ".npy")
            debug_print(f"preparing save_destination for subject {save_dest}")

            if os.path.isfile(save_dest):
                debug_print(f"{save_dest} already exists so no need to load it")
                arr = np.load(save_dest)
            else:
                debug_print(f"{save_dest} does not yet exist so filtering and cutting from scratch")
                raw = mne.io.read_raw_eeglab(eeg_file, preload=True).filter(1, 30).crop(tmin=self.tstart,
                                                                                        tmax=self.tend).drop_channels(
                    ["X", "Y", "Z", "VEOG"])
                raw.set_montage(montage)
                raw.filter(l_freq=0.1, h_freq=100)
                # Segment the data into epochs
                events = mne.make_fixed_length_events(raw, id=1, duration=self.duration, overlap=self.overlap)
                epochs = mne.Epochs(raw, events, tmin=0., tmax=self.duration, baseline=None, preload=True)

                # Apply AutoReject
                ar = AutoReject()
                epochs_clean = ar.fit_transform(epochs)

                arr = epochs_clean.get_data()
                arr = raw.get_data()
                # Apply FASTER
                np.save(save_dest, arr)
                debug_print(f"Preprocessed version saved to {save_dest}")

            self.epochs_list = self.epochs_list or []
            self.epochs_list.append(arr)

            l = self.calc_samples(arr)
            debug_print(f"arr info = {arr.shape}, number of samples= {l}")

            self.data_points = self.data_points or []
            self.data_points.append(l)

            self.y_list.append(y)
            debug_print(f"{subject} has the class {y} meaning its {'PD' if y == 0 else 'CTL'}")

    # Other methods remain unchanged
    def split(self, ratios=0.8, shuffle=False, balance_classes=True):
        if balance_classes:
            # need to know how much one and the other list have
            c1 = [idx for idx, c in enumerate(self.y_list) if c == 1]
            c0 = [idx for idx, c in enumerate(self.y_list) if c == 0]
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
                ce1 = ceil(len(c1) * ratios)
                ce0 = ceil(len(c0) * ratios)
                bottom = c1[:ce1] + c0[:ce0]
                top = c1[ce1:] + c0[ce0:]
            else:
                idx = ceil(len(self.y_list) * ratios)
                bottom = shuffled_idxes[:idx]
                top = shuffled_idxes[idx:]

            return (EEGCwtDataset(self.root_dir, self.participants, self.ids, self.tstart, self.tend, self.special_part,
                                  self.medicated, self.batch_size, use_index=bottom, transform=self.transform,
                                  trans_args=self.trans_args,
                                  overlap=self.overlap, duration=self.duration, debug=False),
                    EEGCwtDataset(self.root_dir, self.participants, self.ids, self.tstart, self.tend, self.special_part,
                                  self.medicated, self.batch_size, use_index=top, transform=self.transform,
                                  trans_args=self.trans_args,
                                  overlap=self.overlap, duration=self.duration, debug=False))
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
                    ce1 = ceil(len(c1) * ratios)
                    ce0 = ceil(len(c0) * ratios)
                    bottom = c1[prev_idx1:ce1] + c0[prev_idx0:ce0]
                else:
                    idx = ceil(len(self.y_list) * ratio)
                    bottom = shuffled_idxes[prev_idx: idx]
                splits.append(
                    EEGCwtDataset(self.root_dir, self.participants, self.ids, self.tstart, self.tend, self.special_part,
                                  self.medicated, self.batch_size, use_index=bottom))
                if balance_classes:
                    prev_idx1 = ce1
                    prev_idx0 = ce0
                else:
                    prev_idx = idx

            if balance_classes:
                bottom = c1[prev_idx1:] + c0[prev_idx0:]
            else:
                bottom = shuffled_idxes[prev_idx:]

            splits.append(
                EEGCwtDataset(self.root_dir, self.participants, self.ids, self.tstart, self.tend, self.special_part,
                              self.medicated, self.batch_size,
                              use_index=bottom))
            return splits


class TestEEGCwtDataset(unittest.TestCase):

    def setUp(self):
        data_path = "ds003490-download"  # Update this path to your dataset
        self.dataset = EEGCwtDataset(data_path,"participants.tsv",debug=True)

    def test_initialization(self):
        self.assertIsNotNone(self.dataset)
        self.assertIsNotNone(self.dataset.epochs_list)
        self.assertIsNotNone(self.dataset.y_list)
        self.assertEqual(len(self.dataset.cwt_cache), len(self.dataset.epochs_list))

    def test_select_channel(self):
        channel = 3
        self.dataset.select_channel(channel)
        self.assertEqual(self.dataset.ch, channel)
        self.assertIsNotNone(self.dataset.cwt_cache[0])  # Assuming at least one epoch in the dataset

    def test_get_item(self):
        channel = 3
        self.dataset.select_channel(channel)
        idx = 5
        cwt_data, label = self.dataset[idx]
        self.assertIsNotNone(cwt_data)
        self.assertIsNotNone(label)
        self.assertEqual(cwt_data.shape[1], self.dataset.duration * self.dataset.freq)

    def test_channel_switching(self):
        channel1 = 3
        channel2 = 5
        idx = 5

        self.dataset.select_channel(channel1)
        cwt_data1, _ = self.dataset[idx]

        self.dataset.select_channel(channel2)
        cwt_data2, _ = self.dataset[idx]

        self.assertNotEqual(channel1, channel2)
        self.assertFalse(np.allclose(cwt_data1, cwt_data2))

    def test_wrong_channel(self):
        with self.assertRaises(ValueError):
            self.dataset.select_channel(-1)

if __name__ == "__main__":
    accuracy = {}
    precision = {}
    recall = {}
    f1 = {}
    batch_size=16
    performance_metrics = {'accuracy': accuracy,
                           'precision': precision,
                           'recall': recall,
                           'f1': f1}

    dset = EEGCwtDataset("ds003490-download", participants="participants.tsv",
                        tstart=0, tend=240, batch_size=8, )
    dtrain, dtest = dset.split()
    del dset

    for i in range(64):
        dtrain.select_channel(i)
        dtest.select_channel(i)

        #node_wise_classification(dtrain,dtest,i,accuracy,precision,recall,f1)

    with open("performance.txt", 'w') as f:
        json.dump(performance_metrics, f, indent=4)

