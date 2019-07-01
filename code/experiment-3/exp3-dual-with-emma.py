"""
    Experiment 3.2: 
        ...
"""

from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from helper import *
from data import *


# ---------------
def train(model, optimizer, criterion, meta):
    loss_curves = []
    X_train, X_valid, y_train, y_valid = get_pulsar_data()
    X_train = apply_corruption(X_train, meta['noise_train'])
    X_valid = apply_corruption(X_valid, meta['noise_valid'])
    for epoch in range(meta['max_epochs']):
        print("Epoch: " + str(epoch+1))
        """ Train """
        model.train()
        loss = train_step(X_train, y_train, model, optimizer, criterion, 
                            meta['batch_size'], train=True)
        """ Validation """
        model.eval()
        with torch.set_grad_enabled(False):
            X_normal, X_ip_noisy, X_snr_noisy, X_all_noisy = split_corruption(X_valid)
            y_normal, y_ip_noisy, y_snr_noisy, y_all_noisy = split_corruption(y_valid)
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

        # sum_loss, n_steps = 0, 0
        # indices = np.arange(X.size(0))
        # np.random.shuffle(indices)
        # for i in range(0, len(X)-batch_size, batch_size):
        #     optimizer.zero_grad()
        #     idx = indices[i:i+batch_size]
        #     batch = X[idx].view(batch_size, -1)
        #     modes = [batch[:,:4], batch[:,4:]]
        #     betas = model[0](modes, train)
        #     new_batch = torch.zeros(128,8).float()
        #     for i in range(128):
        #         new_batch[i,:4] = torch.mul(betas[i,0], batch[i,:4])
        #     new_batch[:,4:] = batch[:,4:]
        #     yhat = model[1](new_batch)  # N x D
        #     if train:
        #         loss = criterion(yhat, y[idx].unsqueeze(-1)) # do regularizer
        #         sum_loss += loss.item()
        #         loss.backward()
        #         optimizer.step()
        #         model[0].cooling()
        #     else:
        #         yhat = yhat >= 0.5
        #         loss = (yhat != y[idx].unsqueeze(-1).byte()).sum().float()/yhat.size(0)
        #         sum_loss += loss.data
        #     n_steps += 1
        # return (sum_loss/n_steps)


# ---------------
if __name__ == "__main__":   
    multi_run = True

    meta = {}
    meta['max_epochs'] = 20
    meta['batch_size'] = 128
    meta['noise_train'] = 5
    meta['noise_valid'] = 5
    criterion = nn.BCELoss()
    model = Model(d_input=8).float()
    optimizer = torch.optim.Adam(nn.ParameterList(model.parameters()))

    if not multi_run:
        model, curves = train(model, optimizer, criterion, meta)
        torch.save(model.state_dict(), "models/exp2-model.pt")
        plot_curves(curves, save=False)
    else:
        loss = {
            'normal': 0,
            'ip-noisy': 0,
            'snr-noisy': 0,
            'all-noisy': 0
            }
        n_runs = 50
        for i in range(n_runs):
            print(i)
            model = Model(d_input=8).float()
            optimizer = torch.optim.Adam(nn.ParameterList(model.parameters()))
            model, _ = train(model, optimizer, criterion, meta)
            _, X_valid, _, y_valid = get_pulsar_data()
            X_valid = apply_corruption(X_valid, meta['noise_valid'])
            X_normal, X_ip_noisy, X_snr_noisy, X_all_noisy = split_corruption(X_valid)
            y_normal, y_ip_noisy, y_snr_noisy, y_all_noisy = split_corruption(y_valid)
            loss['normal'] += train_step(X_normal, y_normal, model, optimizer, 
                            criterion, meta['batch_size'], train=False)
            loss['ip-noisy'] += train_step(X_ip_noisy, y_ip_noisy, model, optimizer, 
                            criterion, meta['batch_size'], train=False)
            loss['snr-noisy'] += train_step(X_snr_noisy, y_snr_noisy, model, optimizer,
                            criterion, meta['batch_size'], train=False)
            loss['all-noisy'] += train_step(X_all_noisy, y_all_noisy, model, optimizer, 
                            criterion, meta['batch_size'], train=False)

        for key, val in loss.items():
            print(key + ": " + str(val/n_runs))
          
    # max_epochs = 20
    # batch_size = 128
    # criterion = nn.BCELoss()
    # model = Model().float()
    # model.load_state_dict(torch.load("model.pt"))
    # model.linear1.weight.data = 2*model.linear1.weight.data
    # autoleft = DenoisingAutoEncoder(4, 12, "sigmoid", 0.1).float()
    # autoleft.load_state_dict(torch.load("autoencoder-left.pt"))
    # autoright = DenoisingAutoEncoder(4, 12, "sigmoid", 0.1).float()
    # autoright.load_state_dict(torch.load("autoencoder-right.pt"))
    # q = [autoleft, autoright]
    # scheduler = LinearScheduler(3, 0.3)
    # qmin = torch.tensor([0, 0]).float()
    # emma = EMMA(2, q, scheduler, qmin).float()
    # total_model = nn.ModuleList([emma, model])

    # optimizer = torch.optim.Adam(nn.ParameterList(model.parameters()))
    # model, curves = train(data, total_model, optimizer, criterion, batch_size, max_epochs)  


















