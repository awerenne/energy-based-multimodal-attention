"""
    ...
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
plt.style.use('seaborn-whitegrid')


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
def true_positives(gt, preds):
    return ((gt == 1) * (preds == 1)).sum()


# ---------------
def false_positives(gt, preds):
    return ((gt == 0) * (preds == 1)).sum()
    

# ---------------
def false_negatives(gt, preds):
    return ((gt == 1) * (preds == 0)).sum()
    

# ---------------
def F1_loss(groundtruth, predictions, threshold):
    predictions = predictions >= threshold
    gt = groundtruth.unsqueeze(-1).byte().clone()
    TP = true_positives(gt, predictions)
    FP = false_positives(gt, predictions)
    FN = false_negatives(gt, predictions)
    precision = TP.float() / (TP+FP).float()
    recall = TP.float() / (TP+FN).float()
    if precision + recall == 0:
        return torch.tensor(0).float(), torch.tensor(0).float(), torch.tensor(0).float()
    f1_score = (2 * precision * recall) / (precision + recall)
    return f1_score, precision, recall


# ---------------
def plot_curves(curves, save=False):
    curves = np.asarray(curves)
    epochs = np.arange(curves.shape[0])
    plt.plot(epochs, curves[:,0], label="Normal")
    plt.plot(epochs, curves[:,1], label="IP-noisy")
    plt.plot(epochs, curves[:,2], label="SNR-noisy")
    plt.plot(epochs, curves[:,3], label="All-noisy")
    plt.legend()
    if save: plt.savefig('results/curves')
    plt.show()




























