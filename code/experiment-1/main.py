"""
    ...
"""


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from sklearn.datasets import make_moons, make_blobs
from sklearn.model_selection import train_test_split

from model import AutoEncoder
from plot import plot_data, plot_vector_field, plot_energy, plot_laplacian
from manifolds import *

seed = 42
batch_size = 32
d_input = 2
n_hidden = 8
max_epochs = 200
n_samples = 1000
noise = 0.08
# noise = 1
type_manifold = 1
with_training = False





# ---------------
def train(loaders, model, optimizer, max_epochs, noise):
    train_loader, test_loader = loaders
    train_curve = []
    test_curve = []
    criterion = nn.MSELoss()
    for epoch in range(max_epochs):
        model.train()
        sum_loss = 0
        n_steps = 0
        for i, X in enumerate(train_loader):
            optimizer.zero_grad()
            X = X[0]
            X_noisy = torch.tensor(X)
            for j in range(X_noisy.size(-1)):
                n = noise * np.random.normal(loc=0.0, scale=1,
                        size=X_noisy.size(0))
                X_noisy[:,j] += torch.tensor(n).float()
            Xhat = model(X_noisy)
            loss = criterion(Xhat, X)
            sum_loss += loss.item()
            n_steps += 1
            loss.backward()
            optimizer.step()
        train_curve.append(sum_loss/n_steps)

        model.eval()
        sum_loss = 0
        n_steps = 0
        for i, X in enumerate(test_loader):
            X = X[0]
            X_noisy = torch.tensor(X)
            for j in range(X_noisy.size(-1)):
                n = noise * np.random.normal(loc=0.0, scale=1,
                        size=X_noisy.size(0))
                X_noisy[:,j] += torch.tensor(n).float()
            Xhat = model(X_noisy)
            loss = criterion(Xhat, X)
            sum_loss += loss.item()
            n_steps += 1
        test_curve.append(sum_loss/n_steps)

        print("Epoch: " + str(epoch))
    return model, (train_curve, test_curve)




# ---------------
if __name__ == "__main__":
    X = make_manifold(n_samples, type_manifold)
    loaders = make_loaders(X)
    if with_training:
        model = AutoEncoder(d_input, n_hidden).float()
        optimizer = torch.optim.Adam(nn.ParameterList(model.parameters()))
        model.train()
        model, curves = train(loaders, model, optimizer, max_epochs, noise)
        plot_curves(curves)
        torch.save(model.state_dict(),"model/autoencoder.pt")
    else:
        model = AutoEncoder(d_input, n_hidden).float()
        model.load_state_dict(torch.load("model/autoencoder.pt"))

    model.eval()
    plot_vector_field(X, model)
    # plot_energy(model)
    # plot_laplacian(model)
    

























