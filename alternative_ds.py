import numpy as np
import torch.cuda
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
import sys
import gc

DEBUG = False

def extract_number(string):
    result = ''.join(filter(str.isdigit, string))
    return int(result) if result else None


def debug_print( message):
    if DEBUG:
        print(f"DEBUG: {message}")


# TODO : implement epoched data and disk usage DONE, so if we are using mmap the cache shouldn't be used at all
class EEGCwtDataset(EEGNpDataset):
    def __init__(self, *args, cwt_file_name = "cwt_quick_access_full", width = 120, cache_len = 1, prep = False, **kwargs):
        self.prep = prep
        self.widths = np.linspace(1, 30, num=width)
        self.cwt_file_name = cwt_file_name
        super().__init__(*args, **kwargs)
        self.cwt_cache = [None] * len(self.epochs_list)
        self.cur_cache_subjects = []
        self.parts = len(self.epochs_list)

        self.cache_len = cache_len

        # self.debug = True

    def clear_cache(self):
        self.cwt_cache = [None] * self.parts

    def apply_cwt(self, data: np.ndarray, channel: int):
        # print(data[channel].shape,)
        if channel == -1:
            # Apply CWT to all channels and stack the results
            print(data.shape)
            cwt_results = [cwt(data[ch], morlet2, self.widths) for ch in range(data.shape[0])]
            cwt_result = np.concatenate(cwt_results, axis=0)
        else:
            cwt_result = cwt(data[channel], morlet2, self.widths)
        return np.array(cwt_result)

    def select_channel(self, channel: int):
        self.clear_cache()
        self.cur_cache_subjects = []
        #self.reset()
        gc.collect()
        #self.load_data()
        print(f"len here : {len(self.epochs_list)}")
        print(channel)
        print(channel is tuple)
        #print(-1 not in channel)
        if type(channel) is tuple and -1 not in channel:
            print("we in")
            self.cwt_cache = [None] * (self.cache_len if not self.disk else  len(self.epochs_list))
            self.ch = channel
            self.dim1 = 512
            #CALCULATE HOW MANY CHANNELS AND THE WIDTHS PER CHANNEL(SIZE OF SPECTROGRAM)
            while True:
                if self.dim1/len(self.ch)%1 == 0:
                    break
                else:
                    self.dim1 = self.dim1 + 1
            width_len = self.dim1/len(self.ch)
            self.widths = np.linspace(1,30, num=int(width_len))
            index = 0
            print(len(self.epochs_list), len(self.subjects))
            while self.epochs_list:
                print(index)
                epoch = self.epochs_list.pop(0)  # Remove and return the first element


                if not os.path.isfile(os.path.join(f"cwts", f"{self.cwt_file_name}_w{len(self.widths)}_{self.subjects.iloc[index].participant_id}_{'c'.join(map(str, self.ch))}.npy")):
                    print(f"{self.cwt_file_name}_w{len(self.widths)}_{self.subjects.iloc[index].participant_id}_{'c'.join(map(str, self.ch))}.npy")
                    hold_me = [np.real(self.apply_cwt(epoch, c)) for c in self.ch]
                    hold_me = np.concatenate(hold_me, axis = 0)
                    np.save(os.path.join(f"cwts", f"{self.cwt_file_name}_w{len(self.widths)}_{self.subjects.iloc[index].participant_id}_{'c'.join(map(str, self.ch))}.npy"), hold_me)

                if self.disk:
                    self.cwt_cache[index] = np.real(np.load(os.path.join(f"cwts", f"{self.cwt_file_name}_w{len(self.widths)}_{self.subjects.iloc[index].participant_id}_{'c'.join(map(str, self.ch))}.npy"), mmap_mode='r'))
                    self.cur_cache_subjects.append(extract_number(self.subjects.iloc[index].participant_id))
                elif len(self.cur_cache_subjects) < self.cache_len:
                    self.cur_cache_subjects.append(extract_number(self.subjects.iloc[index].participant_id))
                    self.cwt_cache[index] = np.load(os.path.join(f"cwts", f"{self.cwt_file_name}_w{len(self.widths)}_{self.subjects.iloc[index].participant_id}_{'c'.join(map(str, self.ch))}.npy"))
                index += 1
        elif channel == -1:
            self.cwt_cache = [None] * (self.cache_len if not self.disk else  len(self.epochs_list))
            self.ch = channel
            index = 0
            print(len(self.epochs_list), len(self.subjects))
            while self.epochs_list:
                print(index)
                epoch = self.epochs_list.pop(0)  # Remove and return the first element

                if not os.path.isfile(os.path.join(f"cwts", f"{self.cwt_file_name}_w{len(self.widths)}_{self.subjects.iloc[index].participant_id}.npy")):
                    print(f"{self.cwt_file_name}_w{len(self.widths)}_{self.subjects.iloc[index].participant_id}.npy")
                    hold_me = np.real(self.apply_cwt(epoch, self.ch))
                    np.save(os.path.join(f"cwts", f"{self.cwt_file_name}_w{len(self.widths)}_{self.subjects.iloc[index].participant_id}.npy"), hold_me)

                if self.disk:
                    self.cwt_cache[index] = np.real(np.load(os.path.join(f"cwts",
                                                                 f"{self.cwt_file_name}_w{len(self.widths)}_{self.subjects.iloc[index].participant_id}.npy"), mmap_mode='r'))
                    #print(self.subjects.iloc[index].participant_id)
                    self.cur_cache_subjects.append(extract_number(self.subjects.iloc[index].participant_id))
                elif len(self.cur_cache_subjects) < self.cache_len:
                    self.cur_cache_subjects.append(extract_number(self.subjects.iloc[index].participant_id))
                    self.cwt_cache[index] = np.load(os.path.join(f"cwts",f"{self.cwt_file_name}_w{len(self.widths)}_{self.subjects.iloc[index].participant_id}.npy"))
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

    def  transfer_to_gpu(self, device):
        """
        Copy the cwt_cache to the specified device.
        """
        self.holder = [torch.tensor(cwt).to(device) for cwt in self.cwt_cache]
        self.cwt_cache, self.holder = self.holder, self.cwt_cache


    def delete_from_gpu(self, device):
        self.cwt_cache = self.holder
        self.holder = []
        torch.cuda.empty_cache()

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
            if self.disk:
                if self.debug:
                    print(f"DEBUG: running on channel {self.ch} and applying CWT from cache")
                    print(f"{self.cwt_cache[idx_subject].shape} cwt shape")
                # print(idx_subject)
                cwt_data = self.cwt_cache[idx_subject][:, idx_inner:(idx_inner + duration)]
                return self.transform(cwt_data, *self.trans_args), self.y_list[idx_subject]
            elif idx_subject in self.cur_cache_subjects:
                innter = self.cur_cache_subjects.index(idx_subject)
                if innter > 0 and self.cur_cache_subjects[-1] < len(self.y_list)-1 and idx_inner > self.batch_size:
                    self.cwt_cache.pop(0)
                    self.cur_cache_subjects.pop(0)
                    hold_me = np.load(os.path.join(f"cwts", f"{self.cwt_file_name}_{self.subjects[idx_subject]}.npy"), mmap_mode='r' if self.disk else None)
                    self.cur_cache_subjects.append(idx_subject)
                    self.cwt_cache.append(hold_me)
                if not self.epoched:
                    return self.transform(self.cwt_cache[innter][:,idx_inner:(idx_inner + duration)], *self.trans_args), self.y_list[idx_subject]
                else:
                    return self.transform(self.cwt_cache[innter][idx_inner, :, :],
                                          *self.trans_args), self.y_list[idx_subject]
            else:
                self.cwt_cache.pop(0)
                self.cur_cache_subjects.pop(0)
                hold_me = np.load(os.path.join(f"cwts", f"{self.cwt_file_name}_{self.subjects[idx_subject]}.npy"), mmap_mode='r' if self.disk else None)
                self.cur_cache_subjects.append(idx_subject)
                self.cwt_cache.append(hold_me)
                return self.transform(self.cwt_cache[-1][:,idx_inner:(idx_inner + duration)], *self.trans_args), self.y_list[idx_subject]
            raise ValueError("A channel must be selected before getting items.")
        else:
            if self.debug:
                print(f"DEBUG: running on channel {self.ch} and applying CWT from cache")
                print(f"{self.cwt_cache[idx_subject].shape} cwt shape")
            # print(idx_subject)
            cwt_data = self.cwt_cache[idx_subject][:, idx_inner:(idx_inner + duration)]
            return self.transform(cwt_data, *self.trans_args), self.y_list[idx_subject]

    def load_data(self):
        #just making the cwt files ready
        if self.prep:
            self.load_data_but_cwt()
            return
        #subjects = pd.read_table(os.path.join(self.root_dir, self.participants))
        f_name = f"len_{self.medicated}_{self.tstart}_{self.tend}_noDrop_{self.overlap}_{self.duration}_np{self.name}".replace(
            '.', 'd')
        fresh_entries = f_name not in self.subjects
        debug_print(f"{f_name} is {'not' if fresh_entries else ''} in subjects")
        self.y_list = []
        self.epochs_list = None
        self.data_points = None
        montage = mne.channels.make_standard_montage('standard_1020')
        cont = 0
        for subject in self.subjects.itertuples():
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

            """if (os.path.isfile(os.path.join(subject_path_eeg,
                                            f"{self.medicated}_{self.tstart}_{self.tend}_noDrop_{self.overlap}_{self.duration}_np_TESTER_clean".replace(
                                                '.', 'd') + ".npy")) and cont >= 48):
                os.remove(os.path.join(subject_path_eeg,
                                       f"{self.medicated}_{self.tstart}_{self.tend}_noDrop_{self.overlap}_{self.duration}_np_TESTER_clean".replace(
                                           '.', 'd') + ".npy"))"""

            cont = cont + 1

            save_dest = os.path.join(subject_path_eeg,
                                     f"{self.medicated}_{self.tstart}_{self.tend}_noDrop_{self.overlap}_{self.duration}_np{self.name}".replace(
                                         '.', 'd') + ".npy")
            debug_print(f"preparing save_destination for subject {save_dest}")

            if os.path.isfile(save_dest):
                debug_print(f"{save_dest} already exists so no need to load it")
                arr = np.load(save_dest, mmap_mode="r" if self.disk else None)
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
                #arr = raw.get_data()
                # Apply FASTER
                print(f"SAVED {len(self.y_list)}")
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
        print(f"I'm lost printing {len(self.epochs_list)} and y len {len(self.y_list)}, {len(self.subjects)}")

    def load_data_but_cwt(self):
        #subjects = pd.read_table(os.path.join(self.root_dir, self.participants))
        f_name = f"{self.cwt_file_name}_w{len(self.widths)}"
        fresh_entries = f_name not in self.subjects
        debug_print(f"{f_name} is {'not' if fresh_entries else ''} in subjects")
        montage = mne.channels.make_standard_montage('standard_1020')

        for subject in self.subjects.itertuples():
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


            save_dest = os.path.join(subject_path_eeg,
                                     f"{self.medicated}_{self.tstart}_{self.tend}_noDrop_{self.overlap}_{self.duration}_np".replace(
                                         '.', 'd') + ".npy")
            eeg_file = os.path.join(subject_path_eeg,
                                    next(f for f in os.listdir(subject_path_eeg) if f.endswith('.set')))
            cwt_save_dest = os.path.join(f"cwts", f"{self.cwt_file_name}_w{len(self.widths)}_{subject.participant_id}.npy")
            debug_print(f"preparing save_destination for CWT transform {cwt_save_dest}")

            if os.path.isfile(cwt_save_dest):
                debug_print(f"{cwt_save_dest} already exists so no need to load it")
                arr = np.load(cwt_save_dest, mmap_mode= 'r' if self.disk else None)
            else:
                if os.path.isfile(save_dest):
                    debug_print(f"{save_dest} already exists so no need to load it")
                    arr = np.load(save_dest, mmap_mode='r' if self.disk else None)
                else:
                    debug_print(f"{cwt_save_dest} does not yet exist so filtering, cutting and transforming from scratch")
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

                # Calculate CWT transform
                arr = self.apply_cwt(arr,-1)

                np.save(cwt_save_dest, arr)
                debug_print(f"CWT transform saved to {cwt_save_dest}")

            self.epochs_list = self.epochs_list or []
            self.epochs_list.append(arr)

            l = self.calc_samples(arr)
            debug_print(f"arr info = {arr.shape}, number of samples= {l}")

            self.data_points = self.data_points or []
            self.data_points.append(l)

            self.y_list.append(y)
            debug_print(f"{subject} has the class {y} meaning its {'PD' if y == 0 else 'CTL'}")
        print(f"We did sth: {len(self.y_list)}")

    # Other methods remain unchanged
    def split(self, ratios=0.8, shuffle=False, balance_classes=True, fractions=False):

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
        if fractions:
            print(self.y_list)
            return [EEGCwtDataset(self.root_dir, self.participants, self.ids, self.tstart, self.tend, self.special_part,
                          self.medicated, self.batch_size, use_index=[i], transform=self.transform,
                          trans_args=self.trans_args,
                          overlap=self.overlap, duration=self.duration, debug=self.debug, width=len(self.widths),
                          cache_len=self.cache_len, disk=self.disk, name=self.name, epoched=self.epoched,
                          prep=self.prep) for i in self.use_index]
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
                                  overlap=self.overlap, duration=self.duration, debug=self.debug, width=len(self.widths),
                                  cache_len=self.cache_len, disk=self.disk, name=self.name, epoched=self.epoched,
                                  prep=self.prep),
                    EEGCwtDataset(self.root_dir, self.participants, self.ids, self.tstart, self.tend, self.special_part,
                                  self.medicated, self.batch_size, use_index=top, transform=self.transform,
                                  trans_args=self.trans_args,
                                  overlap=self.overlap, duration=self.duration, debug=self.debug, width=len(self.widths),
                                  cache_len=self.cache_len, disk=self.disk, name=self.name, epoched=self.epoched,
                                  prep=self.prep))
        else:
            assert isinstance(ratios, tuple)
            splits = []
            if balance_classes:
                prev_idx1 = 0
                prev_idx0 = 0
                debug_print(f"Balancing classes, initialized {prev_idx1} and {prev_idx0} to 0")
            else:
                prev_idx = 0
                debug_print(f"Not balancing classes, initialized {prev_idx} to 0")

            for ratio in ratios:
                debug_print(f"Current ratio to implement: {ratio}")

                if balance_classes:
                    ce1 = ceil(len(c1) * ratio)
                    ce0 = ceil(len(c0) * ratio)
                    debug_print(f"For balanced classes, calculated ce1: {ce1} and ce0: {ce0}")
                    bottom = c1[prev_idx1:prev_idx1 + ce1] + c0[prev_idx0:prev_idx0 + ce0]
                    debug_print(f"Balanced classes bottom indexes: {bottom}")
                else:
                    idx = ceil(len(self.y_list) * ratio)
                    debug_print(f"For unbalanced classes, calculated idx: {idx}")
                    bottom = shuffled_idxes[prev_idx: prev_idx + idx]
                    debug_print(f"Unbalanced classes bottom indexes: {bottom}")

                splits.append(
                    EEGCwtDataset(self.root_dir, self.participants, self.ids, self.tstart, self.tend, self.special_part,
                                  self.medicated, self.batch_size, use_index=bottom, width=len(self.widths),
                                  cache_len=self.cache_len, disk=self.disk, name=self.name, epoched=self.epoched,
                                  prep=self.prep, transform=self.transform, trans_args=self.trans_args,
                                  debug=self.debug)
                )
                debug_print("Added new EEGCwtDataset to splits")

                if balance_classes:
                    prev_idx1 = prev_idx1 + ce1
                    prev_idx0 = prev_idx0 + ce0
                    debug_print(f"Updated prev_idx1: {prev_idx1} and prev_idx0: {prev_idx0}")
                else:
                    prev_idx = prev_idx + idx
                    debug_print(f"Updated prev_idx: {prev_idx}")

            if balance_classes:
                bottom = c1[prev_idx1:] + c0[prev_idx0:]
                debug_print(f"Final balanced classes bottom indexes: {bottom}")
            else:
                bottom = shuffled_idxes[prev_idx:]
                debug_print(f"Final unbalanced classes bottom indexes: {bottom}")

            splits.append(
                EEGCwtDataset(self.root_dir, self.participants, self.ids, self.tstart, self.tend, self.special_part,
                              self.medicated, self.batch_size,
                              use_index=bottom, width=len(self.widths), cache_len=self.cache_len, disk=self.disk,
                              name=self.name, epoched=self.epoched, prep=self.prep, transform=self.transform,
                              trans_args=self.trans_args, )
            )
            debug_print("Added final EEGCwtDataset to splits")

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
    prep = False

    if prep:
        dset = EEGCwtDataset("ds003490-download", participants="participants.tsv",
                             tstart=0, tend=240, batch_size=8, width=8, prep=True,disk=True)
    else:
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
                            tstart=0, tend=240, batch_size=8, epoched = True, disk=True)

        """dtrain, dtest = dset.split()
        del dset

        for i in range(64):
            dtrain.select_channel(i)
            dtest.select_channel(i)

            #node_wise_classification(dtrain,dtest,i,accuracy,precision,recall,f1)

        with open("performance.txt", 'w') as f:
            json.dump(performance_metrics, f, indent=4)

"""
