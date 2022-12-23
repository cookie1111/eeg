from mne.io import read_raw_eeglab
from mne.preprocessing import ICA
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import fastica

if __name__ == '__main__':
    a = read_raw_eeglab("ds003490-download/sub-007/ses-01/eeg/sub-007_ses-01_task-Rest_eeg.set")
    a.plot()


    # Pergorming ICA on the raw signal
    # guidelines below:
    # https://sccn.ucsd.edu/wiki/Makoto%27s_preprocessing_pipeline#General_tips_for_performing_ICA_.2806.2F26.2F2018_updated.29
    ic = ICA(10)
    ic.fit(a)

    ic.plot_sources(a)

    df = ic.to_data_frame()

    print("stand")