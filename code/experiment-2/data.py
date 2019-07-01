"""
    Manipulation of HIGGS dataset.
"""

import torch
import numpy as np
import pandas as pd 


# ---------------
def Generator():
    for chunk in pd.read_csv("../datasets/higgs.csv", header=None, chunksize=1e5):
        X = np.asarray(chunk[range(22,29)].values)  # Discard the high-level features
        y = np.asarray(chunk[0].values)
        X_train = torch.tensor(X[0:int(0.8e5)]).float()
        y_train = torch.tensor(y[0:int(0.8e5)]).float()
        X_valid = torch.tensor(X[int(0.8e5):]).float()
        y_valid = torch.tensor(y[int(0.8e5):]).float()
        yield (X_train, X_valid, y_train, y_valid)


# ---------------
def signal_only(X, y):
    X, y = X.data.numpy(), y.data.numpy()
    _, idx = np.unique(y, return_index=True)
    return torch.tensor(X[idx[1]:]).float()


# ---------------
def background_only(X, y):
    X, y = X.data.numpy(), y.data.numpy()
    _, idx = np.unique(y, return_index=True)
    return torch.tensor(X[idx[0]:idx[1]]).float()


















