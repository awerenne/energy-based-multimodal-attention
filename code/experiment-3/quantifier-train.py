"""
    ...
"""

import numpy as np
import torch
import torch.nn as nn
import sys
import pandas as pd
import matplotlib.pyplot as plt
sys.path.append('../emma/')
from quantifier import DenoisingAutoEncoder
from sklearn.model_selection import train_test_split

from data import signal_only, get_pulsar_data

seed =  42


# ---------------
def train(model, optimizer, criterion, batch_size, max_epochs):
    train_curve, test_curve = [], []
    X_train, X_valid, y_train, y_valid = get_pulsar_data()
    X_train = signal_only(X_train, y_train)
    X_valid = signal_only(X_valid, y_valid)
    for epoch in range(max_epochs):
        print("Epoch: " + str(epoch+1))
        """ Train """
        model.train()
        loss = train_step(X_train[:,4:], model, optimizer, criterion, 
                            batch_size, train=True)
        train_curve.append(loss)

        """ Validation """
        model.eval()
        with torch.set_grad_enabled(False):
            loss = train_step(X_valid[:,4:], model, optimizer, criterion, 
                            batch_size, train=False)
        test_curve.append(loss)
        print("\t\t loss: " + str(loss))
    return model, (train_curve, test_curve)


# ---------------
def train_step(X, model, optimizer, criterion, batch_size, train):
        sum_loss, n_steps = 0, 0
        indices = np.arange(X.size(0))
        np.random.shuffle(indices)
        for i in range(0, len(X)-batch_size, batch_size):
            optimizer.zero_grad()
            idx = indices[i:i+batch_size]
            batch = X[idx].view(batch_size, -1)
            Xhat = model(batch, add_noise=True)  # N x D
            loss = criterion(Xhat, batch)
            sum_loss += loss.item()
            n_steps += 1
            if train:
                loss.backward()
                optimizer.step()
        return (sum_loss/n_steps)

# ---------------
def plot_curves(curves, save=False):
    """ Plot train- and validation curves """
    train_curve, test_curve = curves
    train_curve = np.asarray(train_curve)
    test_curve = np.asarray(test_curve)
    epochs = np.arange(len(train_curve))
    plt.plot(epochs, train_curve, label="Train")
    plt.plot(epochs, test_curve, label="Validation")
    plt.legend()
    plt.show()
    if save: plt.savefig('results/train-valid-curves')
    plt.show()


# ---------------
def get_minimum_energy(model):
    X_train, X_valid, y_train, y_valid = get_pulsar_data()
    model.eval()
    X = signal_only(X_train, y_train)
    q_min = float("Inf")
    for i in range(0, len(X)-1000, 1000):
        q = model.energy(X[i:i+1000,4:])
        if q_min > torch.min(q):
            q_min = torch.min(q).data
    return q_min


# ---------------
if __name__ == "__main__":
    """ Parameters of experiment """
    retrain = True
    max_epochs = 30
    batch_size = 64
    noise = 0.01
    activation = "sigmoid"
    d_input = 4
    n_hidden = 10
    criterion = nn.MSELoss()

    """ Load and train model """ 
    if retrain:
        model = DenoisingAutoEncoder(d_input, n_hidden, activation, noise).float()
        optimizer = torch.optim.Adam(nn.ParameterList(model.parameters()))
        model, curves = train(model, optimizer, criterion, batch_size, max_epochs)
        q_min = get_minimum_energy(model)
        torch.save([model.state_dict(), q_min],"models/q-snr.pt")
        plot_curves(curves)
    else:
        model = DenoisingAutoEncoder(d_input, n_hidden, activation, noise).float()
        q_min = get_minimum_energy(model)
        q, q_min = torch.load("models/q-snr.pt")
        model.load_state_dict(q)
    model.eval()

    print(q_min)
    
    




























