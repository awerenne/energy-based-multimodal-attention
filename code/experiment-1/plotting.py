"""
    ...
"""

import torch
from torch.utils import data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
plt.style.use('seaborn-whitegrid')

seed = 42

# ---------------
def plot_curves(curves, save=False):
    """ Plot train- and validation curves """
    train_curve, valid_curve = curves
    train_curve = np.asarray(train_curve)
    test_curve = np.asarray(valid_curve)
    epochs = np.arange(len(train_curve))
    plt.plot(epochs, train_curve, label="Train")
    plt.plot(epochs, valid_curve, label="Validation")
    plt.legend()
    if save: plt.savefig('results/train-valid-curves')
    plt.show()


# ---------------
def make_mesh():
    nx, ny = (25, 25)
    x = np.linspace(-5, 5, nx)
    y = np.linspace(-5, 5, ny)
    xmesh, ymesh = np.meshgrid(x, y)
    return xmesh, ymesh, nx, ny


# ---------------
def plot_vector_field(model, X, save=False):
    """ Plots the reconstruction vector field alongside the data manifolds """
    xmesh, ymesh, nx, ny = make_mesh()
    xnorm = np.zeros(xmesh.shape)
    ynorm = np.zeros(ymesh.shape)
    for i in range(nx):
        for j in range(ny):
            sample = torch.tensor([xmesh[i,j], ymesh[i,j]]).float().unsqueeze(0)
            reconstruction = model(sample, add_noise=False).squeeze()
            xnorm[i,j] = reconstruction[0] - xmesh[i,j]
            ynorm[i,j] = reconstruction[1] - ymesh[i,j]
    fig1, ax1 = plt.subplots()
    ax1.set_title('Vector field of reconstruction')
    ax1.quiver(xmesh, ymesh, xnorm, ynorm)
    plt.scatter(X[:, 0], X[:, 1], c='blue', edgecolor='k', alpha=0.3)
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
            potential = model.potential(sample)
            if potential < qmin: qmin = potential

    for i in range(nx):
        for j in range(ny):
            sample = torch.tensor([xmesh[i,j], ymesh[i,j]]).float().unsqueeze(0)
            potential = model.potential(sample) - qmin
            q[0,i,j] = torch.log(1e-1 + potential.data)
            q[1,i,j] = -np.log(1e-1) + torch.log(1e-1 + model.reconstruction_norm(sample).data)

    plt.title('Potential heatmap')
    plt.pcolormesh(xmesh, ymesh, q[0,:,:], cmap = 'plasma') 
    plt.colorbar()
    if save: plt.savefig('results/potential')
    plt.show()

    plt.title('Reconstruction norm heatmap')
    plt.pcolormesh(xmesh, ymesh, q[1,:,:], cmap = 'plasma') 
    plt.colorbar()
    if save: plt.savefig('results/reconstruction')
    plt.show()



    

    




























