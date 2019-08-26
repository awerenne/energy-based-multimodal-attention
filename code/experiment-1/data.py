""" 
    Generation of the different two-dimensional manifolds. 
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from torch.autograd import Variable

N = 10000
param_mesh = np.linspace(0, 2*3.1415, N)
seed = 42


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

# ---------------
if __name__ == "__main__":
    X = make_wave(200)
    plt.scatter(X[:, 0], X[:, 1], c='blue', edgecolor='k', alpha=0.3)
    plt.savefig('results/wave-manifold')
    plt.show()



































