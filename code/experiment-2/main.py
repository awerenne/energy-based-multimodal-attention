"""
    ...
"""

import numpy as np
import torch
import torch.nn as nn
import sys
sys.path.append('../emma/')
from autoencoder import DenoisingAutoEncoder
sys.path.append('../data/')
from data import get_pulsar_data, signal_only, background_only, white_noise
from plotting import *


# ---------------
def train(X_train, X_valid, model, optimizer, batch_size, max_epochs):
    train_curve, valid_curve = [], []
    
    for epoch in range(max_epochs):
        print("Epoch: " + str(epoch+1))
        """ Train """
        model.train()
        loss = train_step(X_train, model, optimizer, batch_size, train=True)
        train_curve.append(loss)

        """ Validation """
        model.eval()
        with torch.set_grad_enabled(False):
            loss = train_step(X_valid, model, optimizer, batch_size, train=False)
        valid_curve.append(loss)
        print("\t\t loss: " + str(loss))
    return model, (train_curve, valid_curve)


# ---------------
def train_step(X, model, optimizer, batch_size, train=True):
        sum_loss, n_steps = 0, 0
        criterion = nn.MSELoss()
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
def signal_vs_bckg(X_signal, X_background, model):
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
        q_seen.append(model.potential(x).data)

        x = X_background[i]
        x = x.unsqueeze(0)
        q_unseen.append(model.potential(x).data)
    return [np.asarray(q_seen), np.asarray(q_unseen)]

# ---------------
def new_signal_vs_bckg(X_signal, X_background, model):
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
        q_seen.append(model.potential(x).data)

        x = X_background[i]
        x = x.unsqueeze(0)
        q_unseen.append(model.potential(x).data)
    return [np.asarray(q_seen), np.asarray(q_unseen)]


# ---------------
def missing(X, model):
    """ Extract random sample (size = 100) """
    indices = np.arange(X.size(0))
    np.random.shuffle(indices)
    X = X[indices]

    """ Test """
    q_seen = []
    q_missing = []
    for i in range(100):
        x = X[i]
        x = x.unsqueeze(0)
        q_seen.append(model.potential(x).data)

        x = torch.zeros_like(X[i])
        x = x.unsqueeze(0)
        q_missing.append(model.potential(x).data)
    return [np.asarray(q_seen), np.asarray(q_missing)]


# ---------------
def noisy_signal(X_signal, model):
    """ Extract random sample (size = 100) """
    indices = np.arange(X_signal.size(0))
    np.random.shuffle(indices)
    X_signal = X_signal[indices[:100]]

    """ Test """
    measures = []
    for noise_std in np.linspace(0, 2, 30):
        q = []
        X = X_signal.clone()
        for i in range(X_signal.size(-1)):
            X[:,i] = white_noise(X[:,i].data.numpy(), noise_std).unsqueeze(0)
        potentials = model.potential(X)
        mean_ = torch.mean(potentials).data
        stddev_ = torch.std(potentials).data
        measures.append([noise_std, mean_, stddev_])
    return np.asarray(measures)


# ---------------
if __name__ == "__main__":
    """ Parameters of experiment """
    retrain = True
    max_epochs = 30
    batch_size = 64
    noise_std = 0.01
    d_input = 4
    n_hidden = 12
    
    """ Data """
    train_set, valid_set, test_set = get_pulsar_data("../data/pulsar.csv")

    """ Load and train model """ 
    if retrain:
        X_train, y_train = train_set
        # X_train = signal_only(X_train, y_train)
        X_valid, y_valid = valid_set
        # X_valid = signal_only(X_valid, y_valid)

        X_train_ip, X_valid_ip = X_train[:,:4], X_valid[:,:4] 
        model_ip = DenoisingAutoEncoder(d_input, n_hidden, noise_std).float()
        optimizer = torch.optim.Adam(nn.ParameterList(model_ip.parameters()))
        model, curves_ip = train(X_train_ip, X_valid_ip, model_ip, optimizer,
                                batch_size, max_epochs)
        torch.save(model_ip.state_dict(),"dumps/autoencoder-ip.pt")

        X_train_dm_snr, X_valid_dm_snr = X_train[:,4:], X_valid[:,4:] 
        model_dm_snr = DenoisingAutoEncoder(d_input, n_hidden, noise_std).float()
        optimizer = torch.optim.Adam(nn.ParameterList(model_dm_snr.parameters()))
        model, curves_dm_snr = train(X_train_dm_snr, X_valid_dm_snr, model_dm_snr, optimizer,
                                batch_size, max_epochs)
        torch.save(model_dm_snr.state_dict(),"dumps/autoencoder-dm-snr.pt")
        
        plot_curves(curves_ip, save=True, fname="results/curves-ip")
        plot_curves(curves_dm_snr, save=True, fname="results/curves-dm-snr")

        X_train = signal_only(X_train, y_train)
        X_valid = signal_only(X_valid, y_valid)
        # X_train = background_only(X_train, y_train)
        # X_valid = background_only(X_valid, y_valid)

        X_train_ip, X_valid_ip = X_train[:,:4], X_valid[:,:4] 
        model_ip = DenoisingAutoEncoder(d_input, n_hidden, noise_std).float()
        optimizer = torch.optim.Adam(nn.ParameterList(model_ip.parameters()))
        model, curves_ip = train(X_train_ip, X_valid_ip, model_ip, optimizer,
                                batch_size, max_epochs)
        torch.save(model_ip.state_dict(),"dumps/autoencoder-ip-signal.pt")

        X_train_dm_snr, X_valid_dm_snr = X_train[:,4:], X_valid[:,4:] 
        model_dm_snr = DenoisingAutoEncoder(d_input, n_hidden, noise_std).float()
        optimizer = torch.optim.Adam(nn.ParameterList(model_dm_snr.parameters()))
        model, curves_dm_snr = train(X_train_dm_snr, X_valid_dm_snr, model_dm_snr, optimizer,
                                batch_size, max_epochs)
        torch.save(model_dm_snr.state_dict(),"dumps/autoencoder-dm-snr-signal.pt")
        
        plot_curves(curves_ip, save=True, fname="results/curves-ip-signal")
        plot_curves(curves_dm_snr, save=True, fname="results/curves-dm-snr-signal")
    else:
        model_ip = DenoisingAutoEncoder(d_input, n_hidden, noise_std).float()
        model_ip.load_state_dict(torch.load("dumps/autoencoder-ip-signal.pt"))
        model_dm_snr = DenoisingAutoEncoder(d_input, n_hidden, noise_std).float()
        model_dm_snr.load_state_dict(torch.load("dumps/autoencoder-dm-snr-signal.pt"))
    model_ip.eval()
    model_dm_snr.eval()

    X_test, y_test = test_set
    X_test_ip = X_test[:,:4]
    X_test_dm_snr = X_test[:,4:]
    X_signal = signal_only(X_test, y_test)
    X_bckg = background_only(X_test, y_test)
    X_signal_ip, X_bckg_ip = X_signal[:,:4], X_bckg[:,:4] 
    X_signal_dm_snr, X_bckg_dm_snr = X_signal[:,4:], X_bckg[:,4:] 

    """ Experiment 2.1 """
    # measures_ip = missing(X_test_ip, model_ip)
    # measures_dm_snr = missing(X_test_dm_snr, model_dm_snr)
    # plot_signal_bckg(measures_ip, save=True, fname="results/missing-ip")
    # plot_signal_bckg(measures_dm_snr, save=True, fname="results/missing-dm-snr")

    """ Experiment 2.2 """
    measures_ip = signal_vs_bckg(X_signal_ip, X_bckg_ip, model_ip)
    measures_dm_snr = signal_vs_bckg(X_signal_dm_snr, X_bckg_dm_snr, model_dm_snr)
    # plot_signal_bckg(measures_ip, save=True, fname="results/signal-vs-background-ip")
    # plot_signal_bckg(measures_dm_snr, save=True, fname="results/signal-vs-background-dm-snr")
    # plot_signal_bckg(measures_ip, save=True, fname="results/signal-vs-background-ip-inverse")
    # plot_signal_bckg(measures_dm_snr, save=True, fname="results/signal-vs-background-dm-snr-inverse")

    """ Experiment 2.3 """
    # measures_ip = noisy_signal(X_test_ip, model_ip)
    # measures_dm_snr = noisy_signal(X_test_dm_snr, model_dm_snr)
    # plot_noisy_signal(measures_ip, save=True, fname="results/noisy-signal-ip")
    # plot_noisy_signal(measures_dm_snr, save=True, fname="results/noisy-signal-dm-snr")

    
    
    




























