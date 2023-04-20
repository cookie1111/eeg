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


class EEGCwtDataset(EEGNpDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cwt_cache = [None] * len(self.epochs_list)
        self.parts = len(self.epochs_list)
        self.widths = np.linspace(1, 30, num=60)

    def clear_cache(self):
        self.cwt_cache = [None] * self.parts

    def apply_cwt(self, data: np.ndarray, channel: int):
        #print(data[channel].shape,)
        cwt_result = cwt(data[channel], morlet2, self.widths)
        return np.array(cwt_result)

    def select_channel(self, channel: int):
        del self.cwt_cache
        self.load_data()
        if 0 <= channel < self.epochs_list[0].shape[0]:
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
            idx_subject, idx_inner = self.convert_to_idx(idx)
            if self.debug:
                print(f"DEBUG: subject_number = {idx_subject}\n"
                      f"       sample_index = {idx_inner}")
        except TypeError:
            print(idx, duration)
            print("error encountered something in convert to idx went wrong and the function didn't reach return condition and returned None")
            sys.exit(1)

        if self.ch == -1:
            raise ValueError("A channel must be selected before getting items.")
        else:
            if self.debug:
                print(f"DEBUG: running on channel {self.ch} and applying CWT from cache")
                print(f"{self.cwt_cache.shape} cwt shape")
            cwt_data = self.cwt_cache[idx_subject][:, idx_inner:(idx_inner + duration)]
            return self.transform(cwt_data, *self.trans_args), self.y_list[idx_subject]

    # Other methods remain unchanged
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

            return (EEGCwtDataset(self.root_dir, self.participants, self.ids, self.tstart, self.tend, self.special_part,
                                 self.medicated, self.batch_size, use_index=bottom, transform=self.transform, trans_args=self.trans_args,
                                 overlap=self.overlap, duration=self.duration, debug=False),
                    EEGCwtDataset(self.root_dir, self.participants, self.ids, self.tstart, self.tend, self.special_part,
                                 self.medicated, self.batch_size, use_index=top,transform=self.transform, trans_args=self.trans_args,
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
                    ce1 = ceil(len(c1)*ratios)
                    ce0 = ceil(len(c0)*ratios)
                    bottom = c1[prev_idx1:ce1]+c0[prev_idx0:ce0]
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
                bottom = c1[prev_idx1:]+c0[prev_idx0:]
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

        node_wise_classification(dtrain,dtest,i,accuracy,precision,recall,f1)

    with open("performance.txt", 'w') as f:
        json.dump(performance_metrics, f, indent=4)

