"""
    ...
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Multiple runs

# ---------------
class Model(nn.Module):
    """
    ...
    """

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(8, 4)
        self.linear2 = nn.Linear(4, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        x = torch.sigmoid(x)
        return x

# ---------------
def Generator():
    data = pd.read_csv("../../datasets/pulsar-star/pulsar_stars.csv")
    X, y = np.asarray(data.iloc[:, :8].values), np.asarray(data['target_class'].values)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)
    X_train, X_valid = torch.tensor(X_train).float(), torch.tensor(X_valid).float()
    y_train, y_valid = torch.tensor(y_train).float(), torch.tensor(y_valid).float()
    return (X_train, X_valid, y_train, y_valid)

# ---------------
def train(data, model, optimizer, criterion, batch_size, max_epochs):
    train_curve, test_curve, test_noisy_curve = [], [], []
    X_train, X_valid, y_train, y_valid = data
    X_valid = corruption(X_valid, 5)
    # X_train = corruption(X_train, 5)
    for epoch in range(max_epochs):
        # print("Epoch: " + str(epoch+1))
        """ Train """
        model.train()
        loss = train_step(X_train, y_train, model, optimizer, criterion, 
                            batch_size, train=True)
        train_curve.append(loss)

        """ Validation """
        model.eval()
        with torch.set_grad_enabled(False):
            Q1 = int(X_valid.size(0)/2)
            Q2 = int(X_valid.size(0)/6)
            loss_1 = train_step(X_valid[:Q1], y_valid[:Q1], model, optimizer, criterion, 
                            batch_size, train=False)
            loss_2 = train_step(X_valid[Q1:Q1+Q2], y_valid[Q1:Q1+Q2], model, optimizer, criterion, 
                            batch_size, train=False)
            loss_3 = train_step(X_valid[Q1+Q2:Q1+2*Q2], y_valid[Q1+Q2:Q1+2*Q2], model, optimizer, criterion, 
                            batch_size, train=False)
            loss_4 = train_step(X_valid[Q1+2*Q2:], y_valid[Q1+2*Q2:], model, optimizer, criterion, 
                            batch_size, train=False)
        test_curve.append((loss_1, loss_2, loss_3, loss_4))
    return model, test_curve


# ---------------
def train_step(X, y, model, optimizer, criterion, batch_size, train):
        sum_loss, n_steps = 0, 0
        indices = np.arange(X.size(0))
        np.random.shuffle(indices)
        for i in range(0, len(X)-batch_size, batch_size):
            optimizer.zero_grad()
            idx = indices[i:i+batch_size]
            batch = X[idx].view(batch_size, -1)
            yhat = model(batch)  # N x D
            if train:
                loss = criterion(yhat, y[idx].unsqueeze(-1))
                sum_loss += loss.item()
                loss.backward()
                optimizer.step()
            else:
                yhat = yhat >= 0.5
                loss = (yhat != y[idx].unsqueeze(-1).byte()).sum().float()/yhat.size(0)
                sum_loss += loss.data
            n_steps += 1
        return (sum_loss/n_steps)

# ---------------
def plot_curves(curves, save=False):
    """ Plot train- and validation curves """
    curves = np.asarray(curves)
    epochs = np.arange(curves.shape[0])
    plt.plot(epochs, curves[:,0], label="Normal")
    plt.plot(epochs, curves[:,1], label="Left")
    plt.plot(epochs, curves[:,2], label="Right")
    plt.plot(epochs, curves[:,3], label="Both")
    plt.legend()
    plt.show()
    if save: plt.savefig('results/valid-curves')
    plt.show()

# ---------------
def corruption(X, noise):
    def add_noise(x, noise=0.01):
        x_ = x.data.numpy()
        # noise = np.random.normal(loc=0, scale=stddev, size=np.shape(x_))
        noise = np.random.uniform(low=-noise, high=noise, size=np.shape(x_))
        out = np.add(x_, noise)
        out = torch.from_numpy(out).float()
        return out

    Q1 = int(X.size(0)/2)
    Q2 = int(X.size(0)/6)
    for j in range(X.size(1)):
        if j < 4:
            X[Q1:Q1+Q2,j] = add_noise(X[Q1:Q1+Q2,j], noise)
        else:
            X[Q1+Q2:Q1+2*Q2,j] = add_noise(X[Q1+Q2:Q1+2*Q2,j], noise)
        X[Q1+2*Q2:,j] = add_noise(X[Q1+2*Q2:,j], noise)
    return X

# ---------------
if __name__ == "__main__":
    data = Generator()
    
    max_epochs = 20
    batch_size = 128
    criterion = nn.BCELoss()

    loss_1 = 0
    loss_2 = 0
    loss_3 = 0
    loss_4 = 0
    for i in range(20):
        model = Model().float()
        optimizer = torch.optim.Adam(nn.ParameterList(model.parameters()))
        _, curves = train(data, model, optimizer, criterion, batch_size, max_epochs)
        curves = np.asarray(curves)
        loss_1 += curves[-1,0]
        loss_2 += curves[-1,1]
        loss_3 += curves[-1,2]
        loss_4 += curves[-1,3]
    print(loss_1/20.)
    print(loss_2/20.)
    print(loss_3/20.)
    print(loss_4/20.)



















