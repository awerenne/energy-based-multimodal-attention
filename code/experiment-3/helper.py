"""
    ...
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ---------------
class Model(nn.Module):
    """
        Simple model used for ...
    """

    def __init__(self, d_input):
        super().__init__()
        self.linear1 = nn.Linear(d_input, int(d_input/2))
        self.linear2 = nn.Linear(int(d_input/2), 1)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return torch.sigmoid(x)


# ---------------
def zero_one_loss(groundtruth, predictions):
    n_errors = (predictions != groundtruth.unsqueeze(-1).byte()).sum()
    return n_errors.float()/predictions.size(0)


# ---------------
def plot_curves(curves, save=False):
    curves = np.asarray(curves)
    epochs = np.arange(curves.shape[0])
    plt.plot(epochs, curves[:,0], label="Normal")
    plt.plot(epochs, curves[:,1], label="IP-noisy")
    plt.plot(epochs, curves[:,2], label="SNR-noisy")
    plt.plot(epochs, curves[:,3], label="All-noisy")
    plt.legend()
    plt.show()
    if save: plt.savefig('results/curves')
    plt.show()






























