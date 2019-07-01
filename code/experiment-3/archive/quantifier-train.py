"""
    Experiment II: 

    Train quantifier on a particular class and then compare the quantifier values
    on the seen and unseen classes. Two different experiments are made:

    - Checking how the quantifier value observes on increasingly noisy values 
    (expected: the more noisy, the higher the quantifier value) (Experiment 2.1)

    - Comparing the quantifier on the seen and unseen class (expected: higher engery
    on the unseen class) (Experiment 2.2)
"""

import numpy as np
import torch
import torch.nn as nn
import sys
import pandas as pd
import matplotlib.pyplot as plt
sys.path.append('../../emma/')
from quantifier import DenoisingAutoEncoder
from sklearn.model_selection import train_test_split

seed =  42


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
    train_curve, test_curve = [], []
    X_train, X_valid, y_train, y_valid = data
    for epoch in range(max_epochs):
        print("Epoch: " + str(epoch+1))
        """ Train """
        model.train()
        X_train = signal_only(X_train, y_train)
        loss = train_step(X_train[:,4:8], model, optimizer, criterion, 
                            batch_size, train=True)
        train_curve.append(loss)

        """ Validation """
        model.eval()
        X_valid = signal_only(X_valid, y_valid)
        with torch.set_grad_enabled(False):
            loss = train_step(X_valid[:,4:8], model, optimizer, criterion, 
                            batch_size, train=False)
        test_curve.append(loss)
        print("\t\t loss: " + str(loss))
    return model, (train_curve, test_curve)

# ---------------
def signal_only(X, y):
    X, y = X.data.numpy(), y.data.numpy()
    _, idx = np.unique(y, return_index=True)
    return torch.tensor(X[idx[1]:]).float()

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
if __name__ == "__main__":
    """ Parameters of experiment """
    retrain = True
    max_epochs = 100
    batch_size = 64
    noise = 0.1
    activation = "sigmoid"
    d_input = 4
    n_hidden = 12
    criterion = nn.MSELoss()

    """ Data """
    data = Generator()

    """ Load and train model """ 
    if retrain:
        model = DenoisingAutoEncoder(d_input, n_hidden, activation, noise).float()
        optimizer = torch.optim.Adam(nn.ParameterList(model.parameters()))
        model, curves = train(data, model, optimizer, criterion, batch_size, max_epochs)
        torch.save(model.state_dict(),"autoencoder.pt")
        plot_curves(curves)
    else:
        model = DenoisingAutoEncoder(d_input, n_hidden, activation,
                    noise).float()
        model.load_state_dict(torch.load("autoencoder.pt"))
    model.eval()
    
    




























