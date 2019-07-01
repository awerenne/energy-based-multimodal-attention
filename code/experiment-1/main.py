"""
    Experiment I: train an Autoencoder on simple toy data manifolds. Visualize 
    the reconstruction vector field and energy landscape.
"""

import torch
import torch.nn as nn
from helper import *
from data import *
import sys
sys.path.append('../emma/')
from quantifier import DenoisingAutoEncoder

seed = 42

# ---------------
def train(loaders, model, optimizer, criterion, max_epochs):
    train_loader, test_loader = loaders
    train_curve, test_curve = [], []
    for epoch in range(max_epochs):
        """ Train """
        model.train()
        loss = train_step(train_loader, model, optimizer, criterion, train=True)
        train_curve.append(loss)

        """ Validation """
        model.eval()
        with torch.set_grad_enabled(False):
            loss = train_step(test_loader, model, optimizer, criterion, train=False)
        test_curve.append(loss)
        print("Epoch: " + str(epoch))
    return model, (train_curve, test_curve)


# ---------------
def train_step(loader, model, optimizer, criterion, train):
        sum_loss, n_steps = 0, 0
        for i, X in enumerate(loader):
            optimizer.zero_grad()
            Xhat = model(X[0], add_noise=True)  # N x D
            loss = criterion(Xhat, X[0])
            sum_loss += loss.item()
            n_steps += 1
            if train:
                loss.backward()
                optimizer.step()
        return (sum_loss/n_steps)


# ---------------
if __name__ == "__main__":
    """ Parameters of experiment """
    batch_size = 100
    n_samples = 2000
    max_epochs = 100
    d_input = 2
    n_hidden = 8
    noise = 0.01
    retrain = False
    criterion = nn.MSELoss()
    activation = 'sigmoid'

    """ Manifold """
    # X = make_wave(n_samples)  # N x D (with D = 2)
    X = make_circle(n_samples)  # N x D (with D = 2)
    # X = make_spiral(n_samples)  # N x D (with D = 2)

    """ Load and train model """ 
    loaders = make_loaders(X)
    if retrain:
        model = DenoisingAutoEncoder(d_input, n_hidden, activation, noise).float()
        optimizer = torch.optim.Adam(nn.ParameterList(model.parameters()))
        model, curves = train(loaders, model, optimizer, criterion, max_epochs)
        plot_curves(curves)
        torch.save(model.state_dict(),"dump-models/autoencoder.pt")
    else:
        model = DenoisingAutoEncoder(d_input, n_hidden, activation, noise).float()
        model.load_state_dict(torch.load("dump-models/autoencoder.pt"))

    """ Compare the two quantifiers """ 
    model.eval()
    plot_vector_field(model, X, save=True)
    plot_quantifier(model, save=True)
    

























