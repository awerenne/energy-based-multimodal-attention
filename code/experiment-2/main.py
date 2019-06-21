"""
    ...
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
plt.style.use('seaborn-whitegrid')
from model import AutoEncoder

with_training = False
noise = 0.1
max_epochs = 15
batch_size = 32


# ---------------
def noise_vs_energy():
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


# ---------------
def seen_vs_unseen():
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


# ---------------
if __name__ == "__main__":
    X0_train, X0_valid, X1_valid = torch.load('test_data/zeroandone.pt')
    X0 = (X0_train.float(), X0_valid.float())
    X1_valid = X1_valid.float()

    if with_training:
        model = AutoEncoder(28*28, 1024).float()
        optimizer = torch.optim.Adam(nn.ParameterList(model.parameters()))
        model, curves = train(loader, model, optimizer, max_epochs)
        torch.save(model.state_dict(),"dump-models/autoencoder.pt")
        plot_curves(curves)
    else:
        model = AutoEncoder(28*28, 1024).float()
        model.load_state_dict(torch.load("dump-models/autoencoder.pt"))

    model.eval()
    noise_vs_energy()
    seen_vs_unseen()
    
    




























