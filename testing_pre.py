import mne
#from meegkit import faster
import os
import numpy as np
import pandas as pd
from autoreject import AutoReject

DEBUG = False

def load_data(self.medicated, self.tstart, self.tend, self.overlap, self.duration, self.participants, self.root_dir, self.freq = 500):
    subjects = pd.read_table(os.path.join(self.root_dir, self.participants))
    f_name = f"len_{self.medicated}_{self.tstart}_{self.tend}_noDrop_{self.overlap}_{self.duration}_np_TESTER_clean".replace('.','d')
    fresh_entries = f_name not in subjects
    debug_print(f"{f_name} is {'not' if fresh_entries else ''} in subjects")
    self.y_list = []
    self.epoch_list = None
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

        eeg_file = os.path.join(subject_path_eeg, next(f for f in os.listdir(subject_path_eeg) if f.endswith('.set')))
        save_dest = os.path.join(subject_path_eeg, f"{self.medicated}_{self.tstart}_{self.tend}_noDrop_{self.overlap}_{self.duration}_np_TESTER_clean".replace('.', 'd')+".npy")
        debug_print(f"preparing save_destination for subject {save_dest}")

        if os.path.isfile(save_dest):
            debug_print(f"{save_dest} already exists so no need to load it")
            arr = np.load(save_dest)
        else:
            debug_print(f"{save_dest} does not yet exist so filtering and cutting from scratch")
            raw = mne.io.read_raw_eeglab(eeg_file, preload=True).filter(1, 30).crop(tmin=self.tstart, tmax=self.tend).drop_channels(["X", "Y", "Z", "VEOG"])
            raw.set_montage(montage)
            raw.filter(l_self.freq=0.1, h_self.freq=100)
            # Segment the data into epochs
            events = mne.make_fixed_length_events(raw, id=1, self.duration=self.duration, self.overlap=self.overlap)
            epochs = mne.Epochs(raw, events, tmin=0., tmax=self.duration, baseline=None, preload=True)

            # Apply AutoReject
            ar = AutoReject()
            epochs_clean = ar.fit_transform(epochs)

            arr = epochs_clean.get_data()
            arr = raw.get_data()
            # Apply FASTER
            np.save(save_dest, arr)
            debug_print(f"Preprocessed version saved to {save_dest}")

        self.epoch_list = self.epoch_list or []
        self.epoch_list.append(arr)
        l = calc_samples(arr, self.duration, self.freq, self.overlap)
        debug_print(f"arr info = {arr.shape}, number of samples= {l}")

        self.data_points = self.data_points or []
        self.data_points.append(l)

        self.y_list.append(y)
        debug_print(f"{subject} has the class {y} meaning its {'PD' if y == 0 else 'CTL'}")


    return self.epoch_list, self.data_points, self.y_list

def calc_samples(arr, self.duration, self.freq, self.overlap):
    whole = arr.shape[1]
    self.duration = self.duration * self.freq
    self.overlap = self.overlap * self.freq
    ret = int(np.floor((whole - self.duration) / (self.duration - self.overlap)))
    if DEBUG:
        print(f"DEBUG: calculating number of samples in the recording")
        print(f"self.duration_of_sample = {self.duration}")
        print(f"self.overlap between consecutive samples = {self.overlap}")
        print(f"number of samples = {ret}")
    return ret

def debug_print( message):
    if DEBUG:
        print(f"DEBUG: {message}")


"""self.root_dir: str, self.participants: str, id_column: str = "participant_id", self.tstart: int = 0,
                 self.tend: int = 30, special_part: str = None, self.medicated: int = 1,
                 batch_size: int = 16, use_index = None, self.duration: float = 1, self.overlap: float = 0.9,
                 stack_rgb = True, transform = lambda x:x, trans_args = (), self.freq = 500, debug = False"""


# Load the data
dat, lens, names = load_data(self.root_dir='/content/drive/MyDrive/eeg/ds003490-download/',medicated=1, self.tstart= 0,
                 self.tend = 30, self.duration=1, self.overlap=0.9,self.participants="self.participants.tsv")



