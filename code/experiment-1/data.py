""" 
    Generation of the different two-dimensional manifolds. Each function returns
    a N x D (D = 2) tensor. 
        - N: number of samples
        - D: number of dimensions 
"""

import numpy as np
import matplotlib.pyplot as plt

N = 10000
param_mesh = np.linspace(0, 2*3.1415, N)


# ---------------
def make_wave(n_samples):
    assert n_samples <= N
    samples = param_mesh[np.random.randint(0, N, n_samples)]
    x1 = samples-3
    x2 = np.sin(samples)
    return np.concatenate((np.expand_dims(x1,1), np.expand_dims(x2,1)), axis=1)


# ---------------
def make_circle(n_samples):
    assert n_samples <= N
    samples = param_mesh[np.random.randint(0, N, n_samples)]
    x1 = 3*np.sin(samples)
    x2 = 3*np.cos(samples)
    return np.concatenate((np.expand_dims(x1,1), np.expand_dims(x2,1)), axis=1)


# ---------------
def make_spiral(n_samples):
    assert n_samples <= N
    samples = param_mesh[np.random.randint(0, N, n_samples)]
    r = np.sqrt(samples)
    x1 = r * np.cos(samples)
    x2 = r * np.sin(samples)
    return np.concatenate((np.expand_dims(x1,1), np.expand_dims(x2,1)), axis=1)


# ---------------
def make_loaders(X):
    X_train, X_test = train_test_split(X, test_size=0.33, random_state=seed)
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train).float())
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=16)
    test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test).float())
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=16)
    return (train_loader, test_loader)





































