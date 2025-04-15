import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import scipy.signal as scisig
import scipy.stats
import cvxopt as cv
import cvxopt.solvers

#cvxEDA
def cvxEDA(y, delta, tau0=2., tau1=0.7, delta_knot=10., alpha=8e-4, gamma=1e-2,solver=None, options={}, baseline_correction=2):

    n = len(y)
    rangen = np.arange(n)
    y = cv.matrix(y)

    # bateman ARMA model
    a1 = 1. / min(tau1, tau0)  # a1 > a0
    a0 = 1. / max(tau1, tau0)
    ar = np.array([(a1 * delta + 2.) * (a0 * delta + 2.), 2. * a1 * a0 * delta**2 - 8.,
                   (a1 * delta - 2.) * (a0 * delta - 2.)]) / ((a1 - a0) * delta**2)
    ma = np.array([1., 2., 1.])

    # matrices for ARMA model
    i = np.concatenate((rangen, rangen[1:], rangen[2:]))
    j = np.concatenate((rangen, rangen[:-1], rangen[:-2]))
    A = cv.spmatrix(np.repeat(ar, (n, n - 1, n - 2)), i, j, (n, n))
    M = cv.spmatrix(np.repeat(ma, (n, n - 1, n - 2)), i, j, (n, n))

    # spline
    if np.size(delta_knot) == 1:  # standard usage: delta_knot represents the interval in seconds between knots
        delta_knot_s = int(round(delta_knot / delta))
        knots = np.arange(0, n + delta_knot_s // 2, delta_knot_s)
    else:  # advanced usage: delta_knot represents an array with indices of the spline knots
        knots = np.reshape(delta_knot, -1)
        delta_knot_s = knots[1] - knots[0]
    spl = np.concatenate((np.arange(1., delta_knot_s), np.arange(delta_knot_s, 0., -1.)))  # order 1
    spl = np.convolve(spl, spl, 'full')
    spl /= max(spl)
    # matrix of spline regressors
    i = np.arange(-(len(spl) // 2), (len(spl) + 1) // 2)[:, None] + knots[None, :]
    nB = i.shape[1]
    j = np.tile(np.arange(nB), (len(spl), 1))
    p = np.tile(spl, (nB, 1)).T
    valid = (i >= 0) & (i < n)
    B = cv.spmatrix(p[valid], i[valid], j[valid])

    # baseline correction (0=none, 1=constant, 2=linear, 'spline_offset'=non-penalised spline offset)
    if baseline_correction == 'spline_offset':
        C = B * cv.matrix(1., (nB, 1))
        nC = 1
    else:
        nC = baseline_correction
        C = cv.matrix(1., (n, nC)) if nC < 2 else cv.matrix(np.column_stack((np.ones(n), np.arange(1., n + 1.) / n)))

    # Solve the problem:
    # .5*(M*q + B*l + C*d - y)^2 + alpha*sum(A,1)*q + .5*gamma*l'*l
    # s.t. A*q >= 0

    default_options = {'reltol': 1e-9}
    old_options = cv.solvers.options.copy()
    cv.solvers.options.clear()
    cv.solvers.options.update(default_options)  # set our default options
    cv.solvers.options.update(options)          # allow user options to extend or overwrite default options
    if solver == 'conelp':
        # Use conelp
        z = lambda m, n: cv.spmatrix([], [], [], (m, n))
        G = cv.sparse([[-A, z(2, n), M, z(nB + 2, n)],
                       [z(n + 2, nC), C, z(nB + 2, nC)],
                       [z(n, 1), -1, 1, z(n + nB + 2, 1)],
                       [z(2 * n + 2, 1), -1, 1, z(nB, 1)],
		               [z(n + 2, nB), B, z(2, nB), cv.spmatrix(1., range(nB), range(nB))]])
        h = cv.matrix([z(n, 1), .5, .5, y, .5, .5, z(nB, 1)])
        c = cv.matrix([(cv.matrix(alpha, (1, n)) * A).T, z(nC, 1), 1, gamma, z(nB, 1)])
        res = cv.solvers.conelp(c, G, h, dims={'l': n, 'q': [n + 2, nB + 2], 's': []})
        obj = res['primal objective']
    else:
        # Use qp
        Mt, Ct, Bt = M.T, C.T, B.T
        H = cv.sparse([[Mt * M, Ct * M, Bt * M],
                       [Mt * C, Ct * C, Bt * C],
                       [Mt * B, Ct * B, Bt * B + cv.spmatrix(gamma, range(nB), range(nB))]])
        f = cv.matrix([(cv.matrix(alpha, (1, n)) * A).T - Mt * y, -(Ct * y), -(Bt * y)])
        res = cv.solvers.qp(H, f, cv.spmatrix(-A.V, A.I, A.J, (n, len(f))), cv.matrix(0., (n, 1)), solver=solver)
        obj = res['primal objective'] + .5 * (y.T * y)
    cv.solvers.options.clear()
    cv.solvers.options.update(old_options)

    l = res['x'][-nB:]
    d = res['x'][n : n + nC]
    t = B * l + C * d
    q = res['x'][:n]
    p = A * q
    r = M * q
    e = y - r - t

    return (np.array(a).ravel() for a in (r, p, t, l, d, e, obj))
# E4 (wrist) Sampling Frequencies
fs_dict = {'ACC': 32, 'BVP': 64, 'EDA': 4, 'TEMP': 4, 'label': 700, 'Resp': 700}
WINDOW_IN_SECONDS = 1
label_dict = {'baseline': 1, 'stress': 2, 'amusement': 0}
int_to_label = {1: 'baseline', 2: 'stress', 0: 'amusement'}
feat_names = None
savePath = 'data'
subject_feature_path = '/subject_feats'

if not os.path.exists(savePath):
    os.makedirs(savePath)
if not os.path.exists(savePath + subject_feature_path):
    os.makedirs(savePath + subject_feature_path)

# cvxEDA
def eda_stats(y):
    Fs = fs_dict['EDA']
    yn = (y - y.mean()) / y.std()
    [r, p, t, l, d, e, obj] = cvxEDA(yn, 1. / Fs)
    return [r, p, t, l, d, e, obj]


class SubjectData:

    def __init__(self, main_path, subject_number):
        self.name = f'S{subject_number}'
        self.subject_keys = ['signal', 'label', 'subject']
        self.signal_keys = ['chest', 'wrist']
        self.chest_keys = ['ACC', 'ECG', 'EMG', 'EDA', 'Temp', 'Resp']
        self.wrist_keys = ['ACC', 'BVP', 'EDA', 'TEMP']
        with open(os.path.join(main_path, self.name) + '/' + self.name + '.pkl', 'rb') as file:
            self.data = pickle.load(file, encoding='latin1')
        self.labels = self.data['label']

    def get_wrist_data(self):
        data = self.data['signal']['wrist']
        data.update({'Resp': self.data['signal']['chest']['Resp']})
        return data

    def get_chest_data(self):
        return self.data['signal']['chest']

    def extract_features(self):  # only wrist
        results = {key: get_statistics(self.get_wrist_data()[key].flatten(), self.labels, key)
                    for key in self.wrist_keys}
        return results


# https://github.com/MITMediaLabAffectiveComputing/eda-explorer/blob/master/load_files.py
def butter_lowpass(cutoff, fs, order=5):
    # Filtering Helper functions
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = scisig.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    # Filtering Helper functions
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = scisig.lfilter(b, a, data)
    return y

def get_slope(series):
    linreg = scipy.stats.linregress(np.arange(len(series)), series )
    slope = linreg[0]
    return slope

def get_window_stats(data, label=-1):
    mean_features = np.mean(data)
    std_features = np.std(data)
    min_features = np.amin(data)
    max_features = np.amax(data)

    features = {'mean': mean_features, 'std': std_features, 'min': min_features, 'max': max_features,
                'label': label}
    return features


def get_net_accel(data):
    return (data['ACC_x'] ** 2 + data['ACC_y'] ** 2 + data['ACC_z'] ** 2).apply(lambda x: np.sqrt(x))


def get_peak_freq(x):
    f, Pxx = scisig.periodogram(x, fs=8)
    psd_dict = {amp: freq for amp, freq in zip(Pxx, f)}
    peak_freq = psd_dict[max(psd_dict.keys())]
    return peak_freq



def filterSignalFIR(eda, cutoff=0.4, numtaps=64):
    f = cutoff / (fs_dict['ACC'] / 2.0)
    FIR_coeff = scisig.firwin(numtaps, f)

    return scisig.lfilter(FIR_coeff, 1, eda)


def compute_features(e4_data_dict, labels, norm_type=None):
    # Dataframes for each sensor type
    eda_df = pd.DataFrame(e4_data_dict['EDA'], columns=['EDA'])
    bvp_df = pd.DataFrame(e4_data_dict['BVP'], columns=['BVP'])
    acc_df = pd.DataFrame(e4_data_dict['ACC'], columns=['ACC_x', 'ACC_y', 'ACC_z'])
    temp_df = pd.DataFrame(e4_data_dict['TEMP'], columns=['TEMP'])
    label_df = pd.DataFrame(labels, columns=['label'])
    resp_df = pd.DataFrame(e4_data_dict['Resp'], columns=['Resp'])

    # Filter EDA
    eda_df['EDA'] = butter_lowpass_filter(eda_df['EDA'], 1.0, fs_dict['EDA'], 6)

    # Filter ACM
    for _ in acc_df.columns:
        acc_df[_] = filterSignalFIR(acc_df.values)

    # Adding indices for combination due to differing sampling frequencies
    eda_df.index = [(1 / fs_dict['EDA']) * i for i in range(len(eda_df))]
    bvp_df.index = [(1 / fs_dict['BVP']) * i for i in range(len(bvp_df))]
    acc_df.index = [(1 / fs_dict['ACC']) * i for i in range(len(acc_df))]
    temp_df.index = [(1 / fs_dict['TEMP']) * i for i in range(len(temp_df))]
    label_df.index = [(1 / fs_dict['label']) * i for i in range(len(label_df))]
    resp_df.index = [(1 / fs_dict['Resp']) * i for i in range(len(resp_df))]

    # Change indices to datetime
    eda_df.index = pd.to_datetime(eda_df.index, unit='s')
    bvp_df.index = pd.to_datetime(bvp_df.index, unit='s')
    temp_df.index = pd.to_datetime(temp_df.index, unit='s')
    acc_df.index = pd.to_datetime(acc_df.index, unit='s')
    label_df.index = pd.to_datetime(label_df.index, unit='s')
    resp_df.index = pd.to_datetime(resp_df.index, unit='s')

    # New EDA features
    r, p, t, l, d, e, obj = eda_stats(eda_df['EDA'])
    eda_df['EDA_phasic'] = r
    eda_df['EDA_smna'] = p
    eda_df['EDA_tonic'] = t
        
    # Combined dataframe - not used yet
    df = eda_df.join(bvp_df, how='outer')
    df = df.join(temp_df, how='outer')
    df = df.join(acc_df, how='outer')
    df = df.join(resp_df, how='outer')
    df = df.join(label_df, how='outer')
    df['label'] = df['label'].fillna(method='bfill')
    df.reset_index(drop=True, inplace=True)

    if norm_type =='std':
        # std norm
        df = (df - df.mean()) / df.std()
    elif norm_type =='minmax':
        # minmax norm
        df = (df - df.min()) / (df.max() - df.min())

    # Groupby
    grouped = df.groupby('label')
    baseline = grouped.get_group(1)
    stress = grouped.get_group(2)
    amusement = grouped.get_group(3)
    return grouped, baseline, stress, amusement


def get_samples(data, n_windows, label):
    global feat_names
    global WINDOW_IN_SECONDS

    samples = []
    # Using label freq (700 Hz) as our reference frequency due to it being the largest
    # and thus encompassing the lesser ones in its resolution.
    window_len = fs_dict['label'] * WINDOW_IN_SECONDS

    for i in range(n_windows):
        # Get window of data
        w = data[window_len * i: window_len * (i + 1)]

        # Add/Calc rms acc
        # w['net_acc'] = get_net_accel(w)
        w = pd.concat([w, get_net_accel(w)])
        #w.columns = ['net_acc', 'ACC_x', 'ACC_y', 'ACC_z', 'BVP',
          #           'EDA', 'EDA_phasic', 'EDA_smna', 'EDA_tonic', 'TEMP',
            #         'label']
        cols = list(w.columns)
        cols[0] = 'net_acc'
        w.columns = cols
        
        # Calculate stats for window
        wstats = get_window_stats(data=w, label=label)

        # Seperating sample and label
        x = pd.DataFrame(wstats).drop('label', axis=0)
        y = x['label'][0]
        x.drop('label', axis=1, inplace=True)

        if feat_names ==None:
            feat_names = []
            for row in x.index:
                for col in x.columns:
                    feat_names.append('_'.join([str(row), str(col)]))

        # sample df
        wdf = pd.DataFrame(x.values.flatten()).T
        wdf.columns = feat_names
        wdf = pd.concat([wdf, pd.DataFrame({'label': y}, index=[0])], axis=1)
        
        # More feats
        wdf['BVP_peak_freq'] = get_peak_freq(w['BVP'].dropna())
        wdf['TEMP_slope'] = get_slope(w['TEMP'].dropna())
        samples.append(wdf)

    return pd.concat(samples)


def make_patient_data(subject_id):
    global savePath
    global WINDOW_IN_SECONDS

    # Make subject data object for Sx
    subject = SubjectData(main_path='modelBuilding/dataset/WESAD', subject_number=subject_id)

    # Empatica E4 data - now with resp
    e4_data_dict = subject.get_wrist_data()

    # norm type
    norm_type = None

    # The 3 classes we are classifying
    grouped, baseline, stress, amusement = compute_features(e4_data_dict, subject.labels, norm_type)

    # print(f'Available windows for {subject.name}:')
    n_baseline_wdws = int(len(baseline) / (fs_dict['label'] * WINDOW_IN_SECONDS))
    n_stress_wdws = int(len(stress) / (fs_dict['label'] * WINDOW_IN_SECONDS))
    n_amusement_wdws = int(len(amusement) / (fs_dict['label'] * WINDOW_IN_SECONDS))
    # print(f'Baseline: {n_baseline_wdws}\nStress: {n_stress_wdws}\nAmusement: {n_amusement_wdws}\n')

    #
    baseline_samples = get_samples(baseline, n_baseline_wdws, 1)
    # Downsampling
    # baseline_samples = baseline_samples[::2]
    stress_samples = get_samples(stress, n_stress_wdws, 2)
    amusement_samples = get_samples(amusement, n_amusement_wdws, 0)

    all_samples = pd.concat([baseline_samples, stress_samples, amusement_samples])
    all_samples = pd.concat([all_samples.drop('label', axis=1), pd.get_dummies(all_samples['label'])], axis=1)
    # Selected Features
    # all_samples = all_samples[['EDA_mean', 'EDA_std', 'EDA_min', 'EDA_max',
    #                          'BVP_mean', 'BVP_std', 'BVP_min', 'BVP_max',
    #                        'TEMP_mean', 'TEMP_std', 'TEMP_min', 'TEMP_max',
    #                        'net_acc_mean', 'net_acc_std', 'net_acc_min', 'net_acc_max',
    #                        0, 1, 2]]
    # Save file as csv (for now)
    all_samples.to_csv(f'{savePath}{subject_feature_path}/S{subject_id}_feats_4.csv')

    # Does th==save any space?
    subject = None


def combine_files(subjects):
    df_list = []
    for s in subjects:
        df = pd.read_csv(f'{savePath}{subject_feature_path}/S{s}_feats_4.csv', index_col=0)
        df['subject'] = s
        df_list.append(df)

    df = pd.concat(df_list)

    df['label'] = (df['0'].astype(str) + df['1'].astype(str) + df['2'].astype(str)).apply(lambda x: x.index('1'))
    df.drop(['0', '1', '2'], axis=1, inplace=True)

    df.reset_index(drop=True, inplace=True)

    df.to_csv(f'{savePath}/may14_feats4.csv')

    counts = df['label'].value_counts()
    print('Number of samples per class:')
    for label, number in zip(counts.index, counts.values):
        print(f'{int_to_label[label]}: {number}')


if __name__ == '__main__':

    subject_ids = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]

    for patient in subject_ids:
        print(f'Processing data for S{patient}...')
        make_patient_data(patient)

    combine_files(subject_ids)
    print('Processing complete.')