import os
import pickle
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

# Desired sampling rate for all signals (Hz)
DESIRED_FS = 64  # common rate to preserve PPG detail
WINDOW_IN_SECONDS = 1  # length of each training segment

# E4 original sampling rates
fs_dict = {'BVP': 64, 'EDA': 4, 'TEMP': 4}

class SubjectData:
    def __init__(self, main_path, subject_number):
        self.name = f'S{subject_number}'
        with open(os.path.join(main_path, self.name, f'{self.name}.pkl'), 'rb') as f:
            self.data = pickle.load(f, encoding='latin1')
        # high-frequency labels at 700 Hz
        self.labels = self.data['label']

    def get_raw_wrist(self):
        # returns dict of numpy arrays for BVP, EDA, TEMP
        wrist = self.data['signal']['wrist']
        wrist['TEMP'] = wrist.pop('TEMP')  # rename consistency
        return {k: np.array(wrist[k]).flatten() for k in ['BVP', 'EDA', 'TEMP']}

# -- Helper functions --
def lowpass_filter(signal, fs, cutoff=1.0, order=5):
    b, a = butter(order, cutoff / (0.5 * fs), btype='low')
    return filtfilt(b, a, signal)

# -- Sequence computation --
def compute_raw_sequences(subject, main_path):
    data = SubjectData(main_path, subject).get_raw_wrist()
    n_samples = len(SubjectData(main_path, subject).labels)

    # Build DataFrame at native rates
    dfs = {}
    for key, arr in data.items():
        fs = fs_dict[key]
        time_idx = np.arange(len(arr)) / fs
        df = pd.DataFrame({key: arr}, index=pd.to_datetime(time_idx, unit='s'))
        # basic filtering
        if key == 'EDA':
            df[key] = lowpass_filter(df[key], fs, cutoff=1.0)
        elif key == 'BVP':
            # band‑pass around heartbeat
            df[key] = lowpass_filter(df[key], fs, cutoff=8.0)
        else:
            df[key] = lowpass_filter(df[key], fs, cutoff=0.5)
        dfs[key] = df

    # Resample all to DESIRED_FS
    full = None
    for df in dfs.values():
        r = df.resample(f'{int(1000/DESIRED_FS)}L').interpolate()
        full = r if full is None else full.join(r, how='outer')

    # Build labels at same rate
    label_time = np.arange(len(SubjectData(main_path, subject).labels)) / 700
    label_df = pd.DataFrame({'label': SubjectData(main_path, subject).labels},
                             index=pd.to_datetime(label_time, unit='s'))
    label_resampled = label_df.resample(f'{int(1000/DESIRED_FS)}L').ffill()
    full['label'] = label_resampled['label']

    # Segment into non-overlapping windows
    window_size = DESIRED_FS * WINDOW_IN_SECONDS
    total_windows = int(np.floor(len(full) / window_size))

    X = np.zeros((total_windows, window_size, len(dfs)))
    y = np.zeros((total_windows,), dtype=int)

    for i in range(total_windows):
        segment = full.iloc[i*window_size:(i+1)*window_size]
        X[i] = segment[list(dfs.keys())].values
        # majority vote label
        y[i] = segment['label'].mode()[0]

    return X, y

# -- Main pipeline --
def build_dataset(subject_ids, main_path, out_file='raw_sequences.npz'):
    all_X, all_y = [], []
    for s in subject_ids:
        X, y = compute_raw_sequences(s, main_path)
        all_X.append(X)
        all_y.append(y)
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    np.savez(out_file, X=X, y=y)
    print(f"Saved raw sequence dataset to {out_file}: X.shape={X.shape}, y.shape={y.shape}")

if __name__ == '__main__':
    subjects = [2,3,4,5,6,7,8,9,10,11,13,14,15,16,17]
    build_dataset(subjects, main_path='modelBuilding/dataset/WESAD')
