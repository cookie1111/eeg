from mne.io import read_raw_eeglab, read_raw_brainvision
from mne.preprocessing import ICA
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import fastica

# prve  5 min je rest state z označenim ko so oči odprte in ko so oči zaprte. I guess da referenca??
# pol so pa toni ki se spreminjajo iz sekunde v sekundo, 70% procentov je 440Hz sinusni ton, 15% je 660Hz  sinusni ton,ž
# 15% je naturalistični toni iz zvočnega dataseta.

# zapiranje in odpiranje oči je zelo opazno v EEG posnetku.


if __name__ == '__main__':
    a = read_raw_eeglab("/home/sebastjan/PycharmProjects/eeg/ds003490-download/sub-050/ses-01/eeg/sub-050_ses-01_task-Rest_eeg.set")
    #a = read_raw_brainvision("/home/sebastjan/PycharmProjects/eeg/eeg/Control1025.vhdr")
    a.plot()


    # Pergorming ICA on the raw signal
    # guidelines below:
    # https://sccn.ucsd.edu/wiki/Makoto%27s_preprocessing_pipeline#General_tips_for_performing_ICA_.2806.2F26.2F2018_updated.29
    ic = ICA(30)
    ic.fit(a)


    ic.plot_sources(a)
    ic.plot_components()

    df = a.to_data_frame()

    print(df.head)

    print("stand")