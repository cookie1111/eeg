import mne
import numpy as np
from mne.io import read_raw_eeglab, read_raw_brainvision
from mne.preprocessing import ICA
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import fastica
from eeg_preproc import EEGNpDataset as EEGDataset
from sys import getsizeof
from sklearn.preprocessing import scale, normalize, minmax_scale

# prve  5 min je rest state z označenim ko so oči odprte in ko so oči zaprte. I guess da referenca??
# pol so pa toni ki se spreminjajo iz sekunde v sekundo, 70% procentov je 440Hz sinusni ton, 15% je 660Hz  sinusni ton,ž
# 15% je naturalistični toni iz zvočnega dataseta.

# zapiranje in odpiranje oči je zelo opazno v EEG posnetku.
TEST = 1
if __name__ == '__main__':
    if TEST == 0:
        a = read_raw_eeglab("/home/sebastjan/PycharmProjects/eeg/ds003490-download/sub-050/ses-01/eeg/sub-050_ses-01_task-Rest_eeg.set", preload=True)
        #a = read_raw_brainvision("/home/sebastjan/PycharmProjects/eeg/eeg/Control1025.vhdr")
        low_cut = 0.1
        hi_cut = 30
        a.filter(low_cut, hi_cut)
        a.plot()


        # Pergorming ICA on the raw signal
        # guidelines below:
        # https://sccn.ucsd.edu/wiki/Makoto%27s_preprocessing_pipeline#General_tips_for_performing_ICA_.2806.2F26.2F2018_updated.29
        #ic = ICA(30)
        #ic.fit(a)
        eventados = mne.make_fixed_length_events(a, id=1, duration=2, overlap=1.9)
        epoched = mne.Epochs(a,eventados, event_id=1, preload=True)
        print("len: ",len(epoched))
        print(epoched[0],epoched[1])
        testo = epoched[0:5]
        #ic.plot_sources(a)
        #ic.plot_components()
        print(testo.get_data())
        print(epoched.get_data().shape)
        df = a.to_data_frame()
        #epoched.save('epoch_test.fif')
        #print(df.head)

        print("stand")
    elif TEST == 1:
        ds = EEGDataset("ds003490-download",participants="participants.tsv",tstart=0,tend=240,cache_amount=1, batch_size=8)


        dl = DataLoader()

