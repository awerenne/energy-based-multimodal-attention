"""
    ...
"""

import numpy as np
import torch
import torch.nn as nn
from model import AutoEncoder

retrain = False
max_epochs = 15
batch_size = 100


# ---------------
def noise_energy(X_signal):
    measures = []
    for noise in np.linspace(0, 10, 30):
        q = []
        for i in range(100):
            x = X_signal[i,:]
            x = add_noise(x, noise)
            potential = quantifier(x).energy.data
            recon_error = quantifier(x).reconstruction.data
            q.append([potential, recon_error])
        q = np.asarray(q)
        measures.append([noise, np.mean(q), np.std(q)])
    return np.asarray(measures)


# ---------------
def seen_unseen(X_signal, X_background):
    q_seen = []
    q_unseen = []
    for i in range(100):
        x = X_signal[i]
        potential = quantifier(x).energy.data
        recon_error = quantifier(x).reconstruction.data
        q_seen.append([potential, recon_error])

        x = X_background[i]
        potential = quantifier(x).energy.data
        recon_error = quantifier(x).reconstruction.data
        q_unseen.append([potential, recon_error])
    return np.asarray(q_seen), np.asarray(q_unseen)


# ---------------
if __name__ == "__main__":
    X0_train, X0_valid, X1_valid = torch.load('test_data/zeroandone.pt')
    X0 = (X0_train.float(), X0_valid.float())
    X1_valid = X1_valid.float()

    if retrain:
        model = DenoisingAutoEncoder(21, 128, noise=0.1).float()
        optimizer = torch.optim.Adam(nn.ParameterList(model.parameters()))
        model, curves = train(loader, model, optimizer, max_epochs)
        torch.save(model.state_dict(),"dump-models/autoencoder.pt")
        plot_curves(curves)
    else:
        model = AutoEncoder(21, 128, noise=0.1).float()
        model.load_state_dict(torch.load("dump-models/autoencoder.pt"))

    model.eval()
    m = noise_energy()
    plot_noise_energy(m)
    m = seen_unseen()
    plot_seen_unseen(m)
    
    




























