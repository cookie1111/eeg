import numpy as np
from scipy.signal import cwt, morlet
import os
import mne
from eeg_preproc import EEGNpDataset

class EEGCwtDataset(EEGNpDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cwt_cache = [None] * len(self.epochs_list)
        self.widths = np.arange(1, 31)

    def apply_cwt(self, data: np.ndarray, channel: int):
        cwt_result = cwt(data[channel], morlet, self.widths)
        return np.array(cwt_result)

    def select_channel(self, channel: int):
        if 0 <= channel < self.epochs_list[0].shape[0]:
            self.ch = channel
            for i, epoch in enumerate(self.epochs_list):
                self.cwt_cache = self.apply_cwt(epoch, channel)
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
            cwt_data = self.cwt_cache[idx_subject][:, idx_inner:(idx_inner + duration)]
            return cwt_data, self.y_list[idx_subject]

    # Other methods remain unchanged
