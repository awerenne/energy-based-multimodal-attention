"""
    Functions related to the processing of the Pulsar dataset
"""

import numpy as np
import pandas as pd
import torch
import random
from sklearn.model_selection import train_test_split


# ---------------
def get_pulsar_data(fname):
    data = pd.read_csv(fname)
    data.iloc[:, :8] = standardize(data.iloc[:, :8])
    X, y = np.asarray(data.iloc[:, :8].values), np.asarray(data['target_class'].values)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,
                        random_state=42, stratify=y)
    X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5,
                        random_state=42, stratify=y_test)
    X_train, y_train = torch.tensor(X_train).float(), torch.tensor(y_train).float()
    X_valid, y_valid = torch.tensor(X_valid).float(), torch.tensor(y_valid).float()
    X_test, y_test = torch.tensor(X_test).float(), torch.tensor(y_test).float()
    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)


# ---------------
def standardize(df):
    result = df.copy()
    for feature_name in df.columns:
        mean = df[feature_name].mean()
        std = df[feature_name].std()
        result[feature_name] = (df[feature_name] - mean) / std
    return result


# ---------------    
def white_noise(x, noise_std):
    noise = np.random.normal(loc=0, scale=noise_std, size=np.shape(x))
    out = np.add(x, noise)
    return torch.from_numpy(out).float()


# ---------------
def apply_corruption(X, y, noise_std):
    if noise_std <= 0: return X
    indicator = torch.zeros(X.size(0), 2)
    X_, y_ = X.data.numpy(), y.data.numpy()
    X_ = X_[np.argsort(y_),:]
    y_ = np.sort(y_)
    _, start_idx = np.unique(y_, return_index=True)
    index_first_signal = start_idx[1]
    index_last_signal = X_.shape[0]-1

    size_set = index_last_signal - index_first_signal + 1
    set_indices = set(range(index_first_signal, index_last_signal+1))
    idx_noisy = random.sample(set_indices, int(np.floor(size_set/2)))
    mid = int(len(idx_noisy)/2)
    indicator[idx_noisy[:mid],0] = 1
    indicator[idx_noisy[mid:],1] = 1

    for i in range(4):
        X_[indicator[:,0] == 1, i] = white_noise(X_[indicator[:,0] == 1, i], noise_std)
        X_[indicator[:,1] == 1, 4+i] = white_noise(X_[indicator[:,1] == 1, 4+i], noise_std)
    return torch.tensor(X_).float(), torch.tensor(y_).float(), indicator


# ---------------
def noise_power(noise_std):
    return 10 * np.log10(noise_std**2)


# ---------------
def signal_only(X, y):
    X, y = X.data.numpy(), y.data.numpy()
    X = X[np.argsort(y),:]
    y = np.sort(y)
    _, idx = np.unique(y, return_index=True)
    return torch.tensor(X[idx[1]:]).float()


# ---------------
def background_only(X, y):
    X, y = X.data.numpy(), y.data.numpy()
    X = X[np.argsort(y),:]
    y = np.sort(y)
    _, idx = np.unique(y, return_index=True)
    return torch.tensor(X[idx[0]:idx[1]]).float()




































