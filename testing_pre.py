import mne
#from meegkit import faster
import os
import numpy as np
import pandas as pd
from autoreject import AutoReject
import numpy as np
import matplotlib.pyplot as plt

DEBUG = False

def plot_eeg_data(file_path1, file_path2):
    # Load the data
    eeg_data1 = np.load(file_path1)
    eeg_data2 = np.load(file_path2)

    # Create a time vector
    sample_rate = 500  # Hz
    num_samples = eeg_data1.shape[1]
    time = np.arange(num_samples) / sample_rate
    eeg_data1 = eeg_data1 * 1e5
    eeg_data2 = eeg_data2 * 1e5

    screen_size = plt.rcParams["figure.figsize"]

    # Plot each channel
    plt.figure(figsize=(50, 50))
    for i in range(eeg_data1.shape[0]+ 1):
        plt.plot(time, eeg_data1[i, :], label=f"File1") #+ i*1e6)
        plt.plot(time, eeg_data2[i, :], label=f"File2") #+ i*1e6)
        break

    plt.title(f'Channel {i+1}')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.xlim([0, num_samples/sample_rate])
    # Limit the number of ticks on the x-axis to keep the plot from getting too cluttered
    plt.locator_params(axis='x', nbins=4)

    plt.tight_layout()
    plt.show()

def plot_eeg_signal(npy_file):
    # Load the numpy array from the .npy file
    eeg_signal = np.load(npy_file)

    # Create a time array for the x-axis based on the length of the EEG signal
    time = np.arange(0, len(eeg_signal))

    # Create the plot
    plt.figure(figsize=(10, 4))
    plt.plot(time, eeg_signal)
    plt.title('EEG Signal')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.show()


def load_data(medicated, tstart, tend, overlap, duration, participants, root_dir, freq = 500):
    subjects = pd.read_table(os.path.join(root_dir, participants))
    f_name = f"len_{medicated}_{tstart}_{tend}_noDrop_{overlap}_{duration}_np_TESTER_clean".replace('.','d')
    fresh_entries = f_name not in subjects
    debug_print(f"{f_name} is {'not' if fresh_entries else ''} in subjects")
    y_list = []
    epochs_list = None
    data_points = None
    montage = mne.channels.make_standard_montage('standard_1020')


    for subject in subjects.itertuples():
        subject_path = os.path.join(root_dir, subject.participant_id)
        y = 1 if subject.Group == "CTL" else 0
        debug_print(f"subject is in {'CTL' if y else 'PD'} group")

        if medicated in [0, 2] or y == 1:
            session = "ses-02" if subject.sess1_Med == "OFF" else "ses-01"
        else:
            session = "ses-01" if subject.sess1_Med == "OFF" else "ses-02"
        subject_path = os.path.join(subject_path, session)

        debug_print(f"subject was {'OFF' if medicated == 1 else 'ON'} medication")
        debug_print(f"using only one session per subject")

        subject_path_eeg = os.path.join(subject_path, os.listdir(subject_path)[0])
        debug_print(f"loading subjects eeg from {subject_path_eeg}")

        eeg_file = os.path.join(subject_path_eeg, next(f for f in os.listdir(subject_path_eeg) if f.endswith('.set')))
        save_dest = os.path.join(subject_path_eeg, f"{medicated}_{tstart}_{tend}_noDrop_{overlap}_{duration}_np_TESTER_clean".replace('.', 'd')+".npy")
        debug_print(f"preparing save_destination for subject {save_dest}")

        if os.path.isfile(save_dest):
            debug_print(f"{save_dest} already exists so no need to load it")
            arr = np.load(save_dest)
        else:
            debug_print(f"{save_dest} does not yet exist so filtering and cutting from scratch")
            raw = mne.io.read_raw_eeglab(eeg_file, preload=True).filter(1, 30).crop(tmin=tstart, tmax=tend).drop_channels(["X", "Y", "Z", "VEOG"])
            raw.set_montage(montage)
            raw.filter(l_freq=0.1, h_freq=100)
            # Segment the data into epochs
            events = mne.make_fixed_length_events(raw, id=1, duration=duration, overlap=overlap)
            epochs = mne.Epochs(raw, events, tmin=0., tmax=duration, baseline=None, preload=True)

            # Apply AutoReject
            ar = AutoReject()
            epochs_clean = ar.fit_transform(epochs)

            arr = epochs_clean.get_data()
            arr = raw.get_data()
            # Apply FASTER
            np.save(save_dest, arr)
            debug_print(f"Preprocessed version saved to {save_dest}")

        epochs_list = epochs_list or []
        epochs_list.append(arr)
        l = calc_samples(arr, duration, freq, overlap)
        debug_print(f"arr info = {arr.shape}, number of samples= {l}")

        data_points = data_points or []
        data_points.append(l)

        y_list.append(y)
        debug_print(f"{subject} has the class {y} meaning its {'PD' if y == 0 else 'CTL'}")


    return epochs_list, data_points, y_list

def calc_samples(arr, duration, freq, overlap):
    whole = arr.shape[1]
    duration = duration * freq
    overlap = overlap * freq
    ret = int(np.floor((whole - duration) / (duration - overlap)))
    if DEBUG:
        print(f"DEBUG: calculating number of samples in the recording")
        print(f"duration_of_sample = {duration}")
        print(f"overlap between consecutive samples = {overlap}")
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
#dat, lens, names = load_data(root_dir='/content/drive/MyDrive/eeg/ds003490-download/',medicated=1, tstart= 0,
#                 tend = 30, duration=1, overlap=0.9,participants="self.participants.tsv")


#plot_eeg_data("ds003490-download/sub-001/ses-02/eeg/1_0_240_noDrop_0d9_1_np.npy")
#plot_eeg_data("ds003490-download/sub-001/ses-02/eeg/1_0_240_noDrop_0d9_1_np.npy","ds003490-download/sub-001/ses-02/eeg/1_0_240_noDrop_0d9_1_np_TESTER_clean.npy")
plot_eeg_data("ds003490-download/sub-001/ses-02/eeg/1_0_240_noDrop_0d9_1_np_TESTER_clean.npy","ds003490-download/sub-001/ses-02/eeg/1_0_240_noDrop_0d9_1_np.npy")




