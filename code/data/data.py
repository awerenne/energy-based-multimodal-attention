"""
    Functions related to the processing of the Pulsar dataset
"""

import numpy as np
import pandas as pd
import torch
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
    x_ = x.data.numpy()
    noise = np.random.normal(loc=0, scale=noise_std, size=np.shape(x_))
    out = np.add(x_, noise)
    return torch.from_numpy(out).float()


# ---------------
# def apply_corruption(X, noise_sttdev):
#     if noise_sttdev == 0: return X
#     a = int(X.size(0)/2)
#     b = int(X.size(0)/6)
#     for j in range(X.size(1)):
#         if j < 4:
#             X[a:a+b,j] = add_noise(X[a:a+b,j], noise)
#         else:
#             X[a+b:a+2*b,j] = add_noise(X[a+b:a+2*b,j], noise)
#         X[a+2*b:,j] = add_noise(X[a+2*b:,j], noise)
#     return X


# ---------------
def noise_power(noise_std):
    return 10 * np.log10(noise_std**2)


# ---------------
def split_corruption(X):
    a = int(X.size(0)/2)
    b = int(X.size(0)/6)
    return X[:a], X[a:a+b], X[a+b:a+2*b], X[a+2*b:]


# ---------------
def get_without_noise(N):
    n = torch.ones(N, 2)
    a = int(N/2)
    b = int(N/6)
    n[:a, :] = 0
    n[a:a+b, 0] = -1
    n[a+b:a+2*b, 1] = -1
    n[a+2*b:, :] = 0
    return n


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




































