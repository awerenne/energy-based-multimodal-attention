"""
    Experiment 3.1: 
        Train models on separate modes and analyze results of predictions
        on both normal and noisy data.
"""

from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ---------------
def train(model, optimizer, criterion, meta):
    loss_curves = []
    X_train, X_valid, y_train, y_valid = get_pulsar_data()
    X_valid = corruption(X_valid, meta['noise'])
    for epoch in range(meta['max_epochs']):
        print("Epoch: " + str(epoch+1))
        """ Train """
        model.train()
        loss = train_step(X_train, y_train, model, optimizer, criterion, 
                            meta['batch_size'], train=True)
        """ Validation """
        model.eval()
        with torch.set_grad_enabled(False):
            X_normal, X_ip_noisy, X_snr_noisy, X_all_noisy = split(X_valid)
            y_normal, y_ip_noisy, y_snr_noisy, y_all_noisy = split(y_valid)
            loss_normal = train_step(X_normal, y_normal, model, optimizer, 
                            criterion, meta['batch_size'], train=False)
            loss_ip_noisy = train_step(X_ip_noisy, y_ip_noisy, model, optimizer, 
                            criterion, meta['batch_size'], train=False)
            loss_snr_noisy = train_step(X_snr_noisy, y_snr_noisy, model, optimizer,
                            criterion, meta['batch_size'], train=False)
            loss_all_noisy = train_step(X_all_noisy, y_all_noisy, model, optimizer, 
                            criterion, meta['batch_size'], train=False)
        loss_curves.append((loss_normal, loss_ip_noisy, loss_snr_noisy, loss_all_noisy))
    return model, loss_curves


# ---------------
def train_step(X, y, model, optimizer, criterion, batch_size, train):
        sum_loss, n_steps = 0, 0
        indices = np.arange(X.size(0))
        np.random.shuffle(indices)
        for i in range(0, len(X)-batch_size, batch_size):
            optimizer.zero_grad()
            idx = indices[i:i+batch_size]
            batch = X[idx].view(batch_size, -1)
            yhat = model(batch) 
            if train:
                loss = criterion(yhat, y[idx].unsqueeze(-1))
                sum_loss += loss.item()
                loss.backward()
                optimizer.step()
            else:
                yhat = yhat >= 0.5
                loss = zero_one_loss(y[idx], yhat)
                sum_loss += loss.data
            n_steps += 1
        return (sum_loss/n_steps)


# ---------------
if __name__ == "__main__":   
    meta = {}
    meta['max_epochs'] = 20
    meta['batch_size'] = 128
    meta['noise'] = 5
    criterion = nn.BCELoss()
    model = Model(d_input=4).float()
    optimizer = torch.optim.Adam(nn.ParameterList(model.parameters()))

    model, curves = train(model, optimizer, criterion, meta)
    torch.save(model.state_dict(), "models/exp1-model.pt")
    plot_curves(curves, save=False)