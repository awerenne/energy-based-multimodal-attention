"""
    Manipulation of HIGGS dataset.
"""

import torch
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split


# ---------------
# def Generator():
#     for chunk in pd.read_csv("../datasets/higgs.csv", header=None, chunksize=1e5):
#         X = np.asarray(chunk[range(22,29)].values)  # Discard the high-level features
#         y = np.asarray(chunk[0].values)
#         X_train = torch.tensor(X[0:int(0.8e5)]).float()
#         y_train = torch.tensor(y[0:int(0.8e5)]).float()
#         X_valid = torch.tensor(X[int(0.8e5):]).float()
#         y_valid = torch.tensor(y[int(0.8e5):]).float()
#         yield (X_train, X_valid, y_train, y_valid)


# ---------------
def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        mean = df[feature_name].mean()
        std = df[feature_name].std()
        result[feature_name] = (df[feature_name] - mean) / std
    return result
    

# ---------------
def Generator():
    data = pd.read_csv("../datasets/pulsar.csv")
    data.iloc[:, :8] = normalize(data.iloc[:, :8])
    X, y = np.asarray(data.iloc[:, :8].values), np.asarray(data['target_class'].values)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.33,
                        random_state=42, stratify=y)
    X_train, X_valid = torch.tensor(X_train).float(), torch.tensor(X_valid).float()
    y_train, y_valid = torch.tensor(y_train).float(), torch.tensor(y_valid).float()
    return (X_train, X_valid, y_train, y_valid)


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


















