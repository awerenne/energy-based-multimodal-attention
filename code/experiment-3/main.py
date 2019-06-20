"""
    ...
"""


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from model import AutoEncoder

with_training = True
noise = 0.2
max_epochs = 20


# ---------------
def add_noise(X):
    for j in range(X.size(0)):
        n = noise * np.random.normal(loc=0.0, scale=1, size=X.size(-1))
        X[j,:] += torch.tensor(n).float()
    # plt.imshow(X[0].view(28,28).data.numpy(), cmap='gray')
    # plt.show()
    return X


# ---------------
def train(loaders, model, optimizer, max_epochs):
    train_loader, test_loader = loaders
    train_curve = []
    test_curve = []
    criterion = nn.MSELoss()
    for epoch in range(max_epochs):
        model.train()
        sum_loss = 0
        n_steps = 0
        for i, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            X = data.view(data.size(0), -1)
            X_noisy = add_noise(X)
            Xhat = model(X_noisy)
            loss = criterion(Xhat, X)
            sum_loss += loss.item()
            n_steps += 1
            loss.backward()
            optimizer.step()
            # if i == 20: break
        train_curve.append(sum_loss/n_steps)

        model.eval()
        sum_loss = 0
        n_steps = 0
        for i, (data, target) in enumerate(test_loader):
            X = data.view(data.size(0), -1)
            X_noisy = add_noise(X)
            Xhat = model(X_noisy)
            loss = criterion(Xhat, X)
            sum_loss += loss.item()
            n_steps += 1
            # if i == 10: break
        test_curve.append(sum_loss/n_steps)
        print("Epoch: " + str(epoch))
        print(sum_loss/n_steps)
        print()
    return model, (train_curve, test_curve)


# ---------------
def plot_curves(curves):
    train_curve, test_curve = curves
    train_curve = np.asarray(train_curve)
    test_curve = np.asarray(test_curve)
    epochs = np.arange(len(train_curve))
    plt.plot(epochs, train_curve)
    plt.plot(epochs, test_curve)
    plt.show()


# ---------------
if __name__ == "__main__":

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data_mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data_mnist', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=32, shuffle=True)
    loaders = (train_loader, test_loader)

    if with_training:
        model = AutoEncoder(28*28, 900).float()
        optimizer = torch.optim.Adam(nn.ParameterList(model.parameters()))
        model.train()
        model, curves = train(loaders, model, optimizer, max_epochs)
        torch.save(model.state_dict(),"model/autoencoder.pt")
        plot_curves(curves)
    else:
        model = AutoEncoder(28*28, 1024).float()
        model.load_state_dict(torch.load("model/autoencoder.pt"))

    model.eval()
    for i, (data, target) in enumerate(test_loader):
        X = data.view(data.size(0), -1)
        X_noisy = add_noise(X)
        Xhat = model(X_noisy)
        for i in range(3):
            plt.imshow(X_noisy[i].view(28,28).data.numpy(), cmap='gray')
            plt.show()
            plt.imshow(Xhat[i].view(28,28).data.numpy(), cmap='gray')
            plt.show()
        break
    
    

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from model import AutoEncoder
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
plt.style.use('seaborn-whitegrid')

with_training = False
noise = 0.1
max_epochs = 15
batch_size = 32


# ---------------
def add_noise(X):
    for j in range(X.size(0)):
        n = noise * np.random.normal(loc=0.0, scale=1, size=X.size(-1))
        X[j,:] += torch.tensor(n).float()
    return X


# ---------------
def train(X, model, optimizer, max_epochs):
    X_train, X_valid = X
    train_curve = []
    test_curve = []
    criterion = nn.MSELoss()
    for epoch in range(max_epochs):
        model.train()
        sum_loss = 0
        n_steps = 0
        indices = np.arange(X_train.size(0))
        np.random.shuffle(indices)
        for j in range(0, len(X_train)-batch_size, batch_size):
            optimizer.zero_grad()
            idx = indices[j:j+batch_size]
            sample = X_train[idx].view(batch_size, -1)
            sample_noisy = add_noise(sample)
            preds = model(sample_noisy)
            loss = criterion(preds, sample)
            sum_loss += loss.item()
            n_steps += 1
            loss.backward()
            optimizer.step()
            # if i == 20: break
        train_curve.append(sum_loss/n_steps)

        model.eval()
        sum_loss = 0
        n_steps = 0
        indices = np.arange(X_valid.size(0))
        np.random.shuffle(indices)
        for j in range(0, len(X_valid)-batch_size, batch_size):
            optimizer.zero_grad()
            idx = indices[j:j+batch_size]
            sample = X_valid[idx].view(batch_size, -1)
            sample_noisy = add_noise(sample)
            preds = model(sample_noisy)
            loss = criterion(preds, sample)
            sum_loss += loss.item()
            n_steps += 1
            # if i == 10: break
        test_curve.append(sum_loss/n_steps)

        print("Epoch: " + str(epoch))
    return model, (train_curve, test_curve)


# ---------------
def plot_curves(curves):
    train_curve, test_curve = curves
    train_curve = np.asarray(train_curve)
    test_curve = np.asarray(test_curve)
    epochs = np.arange(len(train_curve))
    plt.plot(epochs, train_curve)
    plt.plot(epochs, test_curve)
    plt.show()


# ---------------
if __name__ == "__main__":
    X0_train, X0_valid, X1_valid = torch.load('test_data/zeroandone.pt')
    X0 = (X0_train.float(), X0_valid.float())
    X1_valid = X1_valid.float()

    # Train on zeros
    if with_training:
        model = AutoEncoder(28*28, 1024).float()
        optimizer = torch.optim.Adam(nn.ParameterList(model.parameters()))
        model.train()
        model, curves = train(X0, model, optimizer, max_epochs)
        torch.save(model.state_dict(),"model/autoencoder.pt")
        plot_curves(curves)
    else:
        model = AutoEncoder(28*28, 1024).float()
        model.load_state_dict(torch.load("model/autoencoder.pt"))

    # Experiments
    model.eval()

    # Exp 4-1: Noise versus energy with bands
    measures = []
    for noise in np.linspace(0, 10, 30):
        energies = []
        for i in range(100):
            X = X0[0][i].view(1,-1) 
            n = noise * np.random.normal(loc=0.0, scale=1, size=X.size(-1))
            X += torch.tensor(n).float()
            # energies.append(model.energy(X).data)
            energies.append((model(X)-X).norm(p=2).pow(2).data/1000000)
        energies = np.asarray(energies)
        measures.append([noise, np.mean(energies), np.std(energies)])
    measures = np.asarray(measures)
    plt.fill_between(measures[:,0], measures[:,1]-2*measures[:,2],
            measures[:,1]+2*measures[:,2])
    plt.plot(measures[:,0], measures[:,1],c='black')
    plt.xlabel('Noise', fontsize=18)
    plt.ylabel('Reconstruction norm', fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=11)
    # plt.show()
    plt.savefig('results/noise_vs_energy')

    # Exp 4-2: seen versus unseen
    plt.figure()
    energies_0 = []
    energies_1 = []
    for i in range(100):
        x = X0[0][i].view(1, -1).data
        energies_0.append((model(x)-x).norm(p=2).pow(2).data/1000000)
        x = X1_valid[i].view(1, -1).data
        energies_1.append((model(x)-x).norm(p=2).pow(2).data/1000000)

        # energies_0.append(model.energy(X0[0][i].view(1, -1)).data)
        # energies_1.append(model.energy(X1_valid[i].view(1, -1)).data)
    plt.boxplot([energies_0, energies_1])
    plt.xticks([1, 2], [0, 1])
    plt.xlabel('Mode', fontsize=18)
    plt.ylabel('Reconstruction norm', fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=11)
    # plt.show()
    plt.savefig('results/seen_vs_unseen')
    




























