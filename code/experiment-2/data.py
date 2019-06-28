"""
    Manipulation of HIGGS dataset.
"""

import torch
import numpy as np
import pandas as pd 


# ---------------
def Generator():
    for chunk in pd.read_csv("../datasets/higgs/HIGGS.csv", header=None, chunksize=1e5):
        X = np.asarray(chunk[range(22,29)].values)  # Discard the high-level features
        y = np.asarray(chunk[0].values)
        X_train = torch.tensor(X[0:int(0.8e5)]).float()
        y_train = torch.tensor(y[0:int(0.8e5)]).float()
        X_valid = torch.tensor(X[int(0.8e5):]).float()
        y_valid = torch.tensor(y[int(0.8e5):]).float()
        yield (X_train, X_valid, y_train, y_valid)


# ---------------
def signal_only(X, y):
    y = y.data.numpy()
    idx = np.argsort(y)
    _, counts_y = np.unique(y, return_counts=True)
    n_zeros = counts_y[0]
    return X[idx[n_zeros:]]


# ---------------
def background_only(X, y):
    y = y.data.numpy()
    idx = np.argsort(y)
    _, counts_y = np.unique(y, return_counts=True)
    n_zeros = counts_y[0]
    return X[idx[:n_zeros]]


















