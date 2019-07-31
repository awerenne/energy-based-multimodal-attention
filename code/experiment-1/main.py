"""
    Experiment I: train an Autoencoder on simple toy data manifolds. Visualize 
    the reconstruction vector field and energy landscape.
"""

import torch
import torch.nn as nn
import sys
sys.path.append('../emma/')
from autoencoder import DenoisingAutoEncoder
from plotting import *
from data import *

seed = 42


# ---------------
def train(loaders, model, optimizer, max_epochs):
    train_loader, test_loader = loaders
    train_curve, test_curve = [], []
    for epoch in range(max_epochs):
        """ Train """
        model.train()
        loss = train_step(train_loader, model, optimizer, train=True)
        train_curve.append(loss)

        """ Validation """
        model.eval()
        with torch.set_grad_enabled(False):
            loss = train_step(test_loader, model, optimizer, train=False)
        test_curve.append(loss)
        print("Epoch: " + str(epoch))
    return model, (train_curve, test_curve)


# ---------------
def train_step(loader, model, optimizer, train):
        sum_loss, n_steps = 0, 0
        criterion = nn.MSELoss()
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
    retrain = False
    batch_size = 100
    n_samples = 2000
    max_epochs = 25
    d_input = 2
    n_hidden = 14  # 8
    noise_stddev = 0.004 # 0.008

    """ Manifold """
    X = make_wave(n_samples)  # N x 2 
    # X = make_circle(n_samples)  # N x 2
    # X = make_spiral(n_samples)  # N x 2 TODO: close-up, more hidden inputs

    """ Load and train model """ 
    loaders = make_loaders(X)
    if retrain:
        model = DenoisingAutoEncoder(d_input, n_hidden, noise_stddev).float()
        optimizer = torch.optim.Adam(nn.ParameterList(model.parameters()))
        model, curves = train(loaders, model, optimizer, max_epochs)
        plot_curves(curves)
        torch.save(model.state_dict(),"dumps/autoencoder.pt")
    else:
        model = DenoisingAutoEncoder(d_input, n_hidden, noise_stddev).float()
        model.load_state_dict(torch.load("dumps/autoencoder.pt"))

    """ Compare the two quantifiers """ 
    model.eval()
    plot_vector_field(model, X, save=False)
    # plot_quantifier(model, save=False)























