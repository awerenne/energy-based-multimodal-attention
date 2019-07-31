"""
    Functions related to the processing of the Pulsar dataset
"""

import numpy as np
import pandas as pd
import torch
import random
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt



# ---------------
def get_pulsar_data(fname):
    data = pd.read_csv(fname)
    X, y = np.asarray(data.iloc[:, :8].values), np.asarray(data['target_class'].values)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,
                        random_state=42, stratify=y)
    X_train, X_test = standardize(X_train, X_test)
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
def standardize(X_train, X_test):
    for i in range(X_train.shape[1]):  # Loop through features
        mean = np.mean(X_train[:,i])
        std = np.std(X_train[:,i])
        X_train[:,i] = (X_train[:,i] - mean)/std
        X_test[:,i] = (X_test[:,i] - mean)/std
    return X_train, X_test


# ---------------    
def white_noise(x, noise_std):
    noise = np.random.normal(loc=0, scale=noise_std, size=np.shape(x))
    out = np.add(x, noise)
    return torch.from_numpy(out).float()


# ---------------
def apply_corruption(X, y, noise_std):
    indicator = torch.zeros(X.size(0), 2)
    if noise_std <= 0: return X, y, indicator
    X_, y_ = X.data.numpy(), y.data.numpy()
    N = X_.shape[0]
    big_mid = int(float(N)/2.)
    small_mid = int(float(big_mid/2))
    indicator[:small_mid,0] = 1
    indicator[small_mid:big_mid,1] = 1
    # for i in range(4):
    #     X_[:small_mid, i] = white_noise(X_[:small_mid, i], noise_std)
    #     X_[small_mid:big_mid, 4+i] = white_noise(X_[small_mid:big_mid, 4+i], noise_std)
    X_[:small_mid, :4] = white_noise(X_[:small_mid, :4], noise_std)
    X_[small_mid:big_mid, 4:] = white_noise(X_[small_mid:big_mid, 4:], noise_std)
    p = np.random.permutation(X_.shape[0])
    return torch.tensor(X_[p,:]).float(), torch.tensor(y_[p]).float(), indicator[p,:]


# ---------------
def signal_only(X, y):
    X, y = X.data.numpy(), y.data.numpy()
    X = X[np.argsort(y),:]
    y = np.sort(y)
    _, idx = np.unique(y, return_index=True)
    return torch.tensor(X[idx[1]:]).float()


# ---------------
def noise_power(noise_std):
    return 10 * np.log10(noise_std**2)


# ---------------
def background_only(X, y):
    X, y = X.data.numpy(), y.data.numpy()
    X = X[np.argsort(y),:]
    y = np.sort(y)
    _, idx = np.unique(y, return_index=True)
    return torch.tensor(X[idx[0]:idx[1]]).float()




































