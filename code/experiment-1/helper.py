"""
    Helper functions. Mostly plotting.
"""

import torch
from torch.utils import data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


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
def make_loaders(X):
    X_train, X_test = train_test_split(X, test_size=0.33, random_state=seed)
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train).float())
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=16)
    test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test).float())
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=16)
    return (train_loader, test_loader)


# ---------------
def make_mesh():
    nx, ny = (40, 40)
    x = np.linspace(-2, 2, nx)
    y = np.linspace(-2, 2, ny)
    xmesh, ymesh = np.meshgrid(x, y)
    return xmesh, ymesh, nx, ny


# ---------------
def plot_vector_field(X, model, save=False):
    """ Plots the reconstruction vector field alongside the data manifolds """
    xmesh, ymesh, nx, ny = make_mesh()
    xnorm = np.zeros(xmesh.shape)
    ynorm = np.zeros(ymesh.shape)
    for i in range(nx):
        for j in range(ny):
            sample = torch.tensor([xmesh[i,j], ymesh[i,j]]).float().unsqueeze(0)
            reconstruction = model(sample).squeeze()
            xnorm[i,j] = reconstruction[0] - xmesh[i,j]
            ynorm[i,j] = reconstruction[1] - ymesh[i,j]
    fig1, ax1 = plt.subplots()
    ax1.set_title('Vector field of reconstruction')
    ax1.quiver(xmesh, ymesh, xnorm, ynorm)
    plt.scatter(X[:, 0], X[:, 1], c='blue', edgecolor='k', alpha=0.3)
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    if save: plt.savefig('results/vector-field')
    plt.show()


# ---------------
def plot_quantifier(model, save=False):
    """ Plots two heatmaps: energy and reconstruction norm """
    xmesh, ymesh, nx, ny = make_mesh()
    q = np.zeros((2, ymesh.shape[0], ymesh.shape[1]))

    qmin = float("inf")
    for i in range(nx):
        for j in range(ny):
            sample = torch.tensor([xmesh[i,j], ymesh[i,j]]).float().unsqueeze(0)
            energy = model.energy(sample)
            if energy < qmin: qmin = energy

    for i in range(nx):
        for j in range(ny):
            sample = torch.tensor([xmesh[i,j], ymesh[i,j]]).float().unsqueeze(0)
            energy = model.energy(sample) - Vmin
            q[0,i,j] = torch.log(1e-1 + energy.data)
            q[1,i,j] = torch.log(1e-1 + model.reconstruction(sample).data)

    plt.title('Energy heatmap')
    plt.pcolormesh(xmesh, ymesh, q[0,:,:], cmap = 'plasma') 
    plt.colorbar()
    if save: plt.savefig('results/quantifier-energy')
    plt.show()

    plt.title('Reconstruction heatmap')
    plt.pcolormesh(xmesh, ymesh, q[0,:,:], cmap = 'plasma') 
    plt.colorbar()
    if save: plt.savefig('results/quantifier-reconstruction')
    plt.show()



    

    




























