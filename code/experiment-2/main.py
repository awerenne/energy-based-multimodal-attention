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
from helper import *
from data import *
import sys
sys.path.append('../emma/')
from quantifier import DenoisingAutoEncoder

seed =  42

# ---------------
def noise_quantifier(generator, quantifier):
    """ Test on validation set on the signal class """
    _, X_valid, _, y_valid = next(generator)
    X_signal = signal_only(X_valid, y_valid)

    """ Extract random sample (size = 100) """
    indices = np.arange(X_signal.size(0))
    np.random.shuffle(indices)
    X_signal = X_signal[indices]

    """ Test """
    measures = []
    for noise in np.linspace(0, 10, 30):
        q = []
        for i in range(100):
            x = X_signal[i,:]
            x = add_noise(x, 10**noise)
            x = x.unsqueeze(0)
            potential = quantifier.energy(x).data
            recon_error = quantifier.reconstruction(x, add_noise=False).data
            q.append([potential, recon_error])
        q = np.asarray(q)
        measures.append([[noise, noise], np.mean(q, axis=0), np.std(q, axis=0)])
    return np.asarray(measures)


# ---------------
def seen_unseen(generator, quantifier):
    """ Test on validation set on the signal class """
    _, X_valid, _, y_valid = next(generator)
    X_signal = signal_only(X_valid, y_valid)
    X_background = background_only(X_valid, y_valid)

    """ Extract random sample (size = 100) """
    indices = np.arange(X_signal.size(0))
    np.random.shuffle(indices)
    X_signal = X_signal[indices]
    indices = np.arange(X_background.size(0))
    np.random.shuffle(indices)
    X_background = X_background[indices]

    """ Test """
    q_seen = []
    q_unseen = []
    for i in range(100):
        x = X_signal[i]
        x = x.unsqueeze(0)
        potential = quantifier.energy(x).data
        recon_error = quantifier.reconstruction(x, add_noise=False).data
        q_seen.append([potential, recon_error])

        x = X_background[i]
        x = x.unsqueeze(0)
        potential = quantifier.energy(x).data
        recon_error = quantifier.reconstruction(x, add_noise=False).data
        q_unseen.append([potential, recon_error])
    return [np.asarray(q_seen), np.asarray(q_unseen)]


# ---------------
def train(generator, model, optimizer, criterion, batch_size, max_epochs):
    train_curve, test_curve = [], []
    for epoch in range(max_epochs):
        print("Epoch: " + str(epoch+1))
        for (X_train, X_valid, y_train, y_valid) in generator:
            """ Train """
            model.train()
            X_train = signal_only(X_train, y_train)
            loss = train_step(X_train, model, optimizer, criterion, 
                                batch_size, train=True)
            train_curve.append(loss)

            """ Validation """
            model.eval()
            X_valid = signal_only(X_valid, y_valid)
            with torch.set_grad_enabled(False):
                loss = train_step(X_valid, model, optimizer, criterion, 
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
if __name__ == "__main__":
    """ Parameters of experiment """
    retrain = False
    max_epochs = 1
    batch_size = 256
    noise = 0.1
    activation = "sigmoid"
    d_input = 7
    n_hidden = 32
    criterion = nn.MSELoss()

    """ Data """
    generator = Generator()

    """ Load and train model """ 
    if retrain:
        model = DenoisingAutoEncoder(d_input, n_hidden, activation, noise).float()
        optimizer = torch.optim.Adam(nn.ParameterList(model.parameters()))
        model, curves = train(generator, model, optimizer, criterion, batch_size, max_epochs)
        torch.save(model.state_dict(),"dump-models/autoencoder.pt")
        plot_curves(curves)
    else:
        model = DenoisingAutoEncoder(d_input, n_hidden, activation,
                    noise).float()
        model.load_state_dict(torch.load("dump-models/autoencoder.pt"))
    model.eval()

    """ Experiment 2.1 """
    # measures = noise_quantifier(generator, model)
    # plot_noise_quantifier(measures, save=True)

    """ Experiment 2.2 """
    measures = seen_unseen(generator, model)
    plot_seen_unseen(measures, save=True)
    
    




























