from mne.io import read_raw_eeglab
import matplotlib.pyplot as plt

if __name__ == '__main__':
    a = read_raw_eeglab("/home/sebastjan/Downloads/ds003490-download/sub-001/ses-01/eeg/sub-001_ses-01_task-Rest_eeg.set")
    print("loaded")
    #a.plot(block=False)
    channel = a[0,:]
    #plt.ion()