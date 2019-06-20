"""
    ...
"""

import torch
import torch.nn as nn
from model import AutoEncoder
from helper import *
from manifolds import *

seed = 42
batch_size = 32
d_input = 2
n_hidden = 8
max_epochs = 200
n_samples = 1000
noise = 0.08
retrain = False

# TODO: add generic noise process in inference 
# TODO: early stopping


# ---------------
def train(loaders, model, optimizer, max_epochs, noise):
    train_loader, test_loader = loaders
    train_curve = []
    test_curve = []
    criterion = nn.MSELoss()
    for epoch in range(max_epochs):
        model.train()
        loss = inference(train_loader, model, optimizer, noise, train=true)
        train_curve.append(loss)

        model.eval()
        with torch.set_grad_enabled(False):
            loss = inference(test_loader, model, optimizer, noise, train=false)
        test_curve.append(loss)
        print("Epoch: " + str(epoch))
    return model, (train_curve, test_curve)


# ---------------
def inference(loader, model, optimizer, noise, train):
        sum_loss = 0
        n_steps = 0
        for i, X in enumerate(loader):
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
            if train:
                loss.backward()
                optimizer.step()
        return (sum_loss/n_steps)


# ---------------
if __name__ == "__main__":
    X = make_wave(n_samples)
    loaders = make_loaders(X)
    if retrain:
        model = AutoEncoder(d_input, n_hidden).float()
        optimizer = torch.optim.Adam(nn.ParameterList(model.parameters()))
        model, curves = train(loaders, model, optimizer, max_epochs, noise)
        plot_curves(curves)
        torch.save(model.state_dict(),"trained-models/autoencoder.pt")
    else:
        model = AutoEncoder(d_input, n_hidden).float()
        model.load_state_dict(torch.load("trained-models/autoencoder.pt"))

    model.eval()
    plot_vector_field(model, X)
    plot_quantifier(model)
    

























