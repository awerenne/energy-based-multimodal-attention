"""
    ...
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys
sys.path.append('../emma/')
from autoencoder import DenoisingAutoEncoder
from emma import EMMA, WeightClipper
sys.path.append('../data/')
from data import *
from plotting import *
import copy


# ---------------
class Model(nn.Module):
    """
        ...
    """

    def __init__(self, d_input):
        super().__init__()
        self.linear1 = nn.Linear(d_input, int(d_input/2))
        self.linear2 = nn.Linear(int(d_input/2), 1)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return torch.sigmoid(x)


# ---------------
def copy_emma_model(model_original):
    model = Model(d_input=np.sum(d_input)).float()
    emma = EMMA(model_original[0].n_modes, model_original[0].autoencoders,
            model_original[0].min_potentials).float()
    model = nn.ModuleList([emma, model])
    model.load_state_dict(model_original.state_dict())
    return model


# ---------------
def train_autoencoder(model, optimizer, batch_size, n_epochs, X_train, X_valid=None):
    valid_curve = []
    for epoch in range(n_epochs):
        model.train()
        train_step_autoencoder(X_train, model, optimizer, batch_size, train=True)
        if not X_valid is None:
            model.eval()
            with torch.set_grad_enabled(False):
                loss = train_step_autoencoder(X_valid, model, optimizer, batch_size, train=False)
            valid_curve.append(loss)
    return model, valid_curve


# ---------------
def train_step_autoencoder(X, model, optimizer, batch_size, train=True):
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
def get_min_potential(X, model):
    min_potential = float("Inf")
    indices = np.arange(X.size(0))
    batch_size = 32
    for i in range(0, len(X)-batch_size, batch_size):
        optimizer.zero_grad()
        idx = indices[i:i+batch_size]
        batch = X[idx].view(batch_size, -1)
        potentials = model.potential(batch)
        if min_potential > torch.min(potentials).data:
            min_potential = torch.min(potentials.data)
    return min_potential


# ---------------
def train_clf(model_original, optimizer, meta, X_train, y_train, X_valid=None,
        y_valid=None, with_emma=False):
    n_epochs = meta['max_epochs']
    if not with_emma:
        measures = torch.zeros(n_epochs, 2, requires_grad=False)
        for epoch in range(n_epochs):
            model_original.train()
            train_step_clf(X_train, y_train, model_original, optimizer, meta, train=True)
            if (not X_valid is None) and (not y_valid is None):
                model_original.eval()
                with torch.set_grad_enabled(False):
                    F1, best_threshold = train_step_clf(X_valid, y_valid, model_original,
                            optimizer, meta, train=False)
                    measures[epoch, 0] = F1
                    measures[epoch, 1] = best_threshold
        print(measures[n_epochs-1, 0])
        print(measures[n_epochs-1, 1])
        return model_original, measures
    
    coldness = meta['coldness']
    lambdas = meta['lambda']
    measures = torch.zeros(len(coldness), len(lambdas), n_epochs, 2,
                    requires_grad=False)
    for i, tau in enumerate(coldness):
        for j, lambda_ in enumerate(lambdas):
            model = copy_emma_model(model_original)
            model[0].set_coldness(tau)
            for epoch in range(n_epochs):
                model.train()
                train_step_clf(X_train, y_train, model, optimizer, meta, True, True, lambda_)
                if (not X_valid is None) and (not y_valid is None):
                    model.eval()
                    with torch.set_grad_enabled(False):
                        F1, best_threshold = train_step_clf(X_valid, y_valid, model,
                                optimizer, meta, train=False, with_emma=True, lambda_=lambda_)
                        measures[i, j, epoch, 0] = F1
                        measures[i, j, epoch, 1] = best_threshold
            print(measures[i, j, n_epochs-1, 0])
            print(measures[i, j, n_epochs-1, 1])
    return model, measures


# ---------------
def train_step_clf(X, y, model, optimizer, meta, train=True, with_emma=False,
        lambda_=None):
    n_steps = 0
    thresholds = np.linspace(0.1,0.9,9)
    F1 = np.zeros(11)
    batch_size = meta['batch_size']
    indices = np.arange(X.size(0))
    np.random.shuffle(indices)
    for i in range(0, len(X)-batch_size, batch_size):
        optimizer.zero_grad()
        idx = indices[i:i+batch_size]
        batch = X[idx].view(batch_size, -1)

        if not with_emma:
            yhat = model(batch)
        else:
            modes = [batch[:,:4], batch[:,4:]]
            modes, log_alphas = model[0](modes, train)
            new_batch = torch.zeros(batch.size()).float()
            new_batch[:, :4], new_batch[:, 4:]  = modes[0], modes[1]
            yhat = model[1](new_batch)  # N x D

        if train:
            if not with_emma:
                loss = meta['criterion'](yhat, y[idx].unsqueeze(-1))
            else:
                loss = meta['criterion'](yhat, y[idx].unsqueeze(-1))
                # loss = meta['criterion'](yhat, y[idx].unsqueeze(-1)) + \
                #             regularizer(without_noise[idx], log_alphas)
            loss.backward()
            optimizer.step()
            if with_emma:
                model[0].apply(meta['clipper'])
        else:
            for j, threshold in enumerate(thresholds):
                f1, _, _ = compute_F1(y[idx], yhat, threshold)
                F1[j] += f1
        n_steps += 1
    if not train:
        F1 /= n_steps
        idx = np.argmax(F1)
        return F1[idx], thresholds[idx]


# ---------------
def model_evaluation():
    pass


# ---------------
def compute_F1(y, yhat, threshold):
    yhat = yhat >= threshold
    y = y.unsqueeze(-1).byte().clone()
    TP = true_positives(y, yhat)
    FP = false_positives(y, yhat)
    FN = false_negatives(y, yhat)
    if TP + FP == 0: precision = 0
    else: precision = TP.float() / (TP+FP).float()
    if TP + FN == 0: recall = 0
    else: recall = TP.float() / (TP+FN).float()
    if precision + recall == 0:
        return torch.tensor(0).float(), torch.tensor(0).float(), torch.tensor(0).float()
    f1_score = (2 * precision * recall) / (precision + recall)
    return f1_score, precision, recall


# ---------------
def true_positives(y, yhat):
    return ((y == 1) * (yhat == 1)).sum()


# ---------------
def false_positives(y, yhat):
    return ((y == 0) * (yhat == 1)).sum()
    

# ---------------
def false_negatives(y, yhat):
    return ((y == 1) * (yhat == 0)).sum()


# ---------------
def freeze(model):
    for name, param in model.named_parameters():
        param.requires_grad = False


# ---------------
def unfreeze(model):
    for name, param in model.named_parameters():
        param.requires_grad = True

# TODO
    # Do corruptions in retrain!
    # Regularizer
    # Check EMMA is well reloaded when training a second time (train+valid)
    # Plot a written evaluation from returned matrix

# ---------------
if __name__ == "__main__":  
    """ Parameters experiment """
    retrain = True
    n_modes = 2
    d_input = [4, 4]
    n_hidden = 12  # Number of hidden units in autoencoders
    noise_std_autoenc = 0.01
    noise_std_data = 1 
    coldness = [0, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e2]
    lambda_ = np.linspace(0, 2, 19)

    meta = {}
    meta['criterion'] = nn.BCELoss()
    meta['clipper'] = WeightClipper()
    meta['coldness'] = coldness
    meta['lambda'] = lambda_
    meta['max_epochs'] = 20
    meta['batch_size'] = 128

    """ Data """
    train_set, valid_set, test_set = get_pulsar_data("../data/pulsar.csv")
    X_train, y_train = train_set
    X_valid, y_valid = valid_set
    X_test, y_test = test_set

    """ Training """
    autoencoders = {'integrated-profile': None, 'DM-SNR': None}
    models = {'base-model': None, 'model-without': None, 'model-with': None}
    min_potentials = [None, None]
    if retrain:
        """ Train autoencoders on normal signal train-set """
        X_signal_train = signal_only(X_train, y_train)
        X_signal_valid = signal_only(X_valid, y_valid)
        model = DenoisingAutoEncoder(d_input[0], n_hidden, noise_std_autoenc).float()
        optimizer = torch.optim.Adam(nn.ParameterList(model.parameters()))
        autoencoders['integrated-profile'], _ = train_autoencoder(model, optimizer,
                64, 30, X_signal_train[:,:4], X_signal_valid[:,:4])
        min_potentials[0] = get_min_potential(X_signal_train[:,:4], model)

        model = DenoisingAutoEncoder(d_input[1], n_hidden, noise_std_autoenc).float()
        optimizer = torch.optim.Adam(nn.ParameterList(model.parameters()))
        autoencoders['DM-SNR'], _ = train_autoencoder(model, optimizer, 64, 30,
                X_signal_train[:,4:], X_signal_valid[:,4:])
        min_potentials[1] = get_min_potential(X_signal_train[:,4:], model)
        min_potentials = torch.tensor(min_potentials).float()

        """ Freeze autoencoders """
        for key, model in autoencoders.items():
            freeze(model)

        """ Train base model on normal train-set, eval on noisy valid-set """
        model = Model(d_input=np.sum(d_input)).float()
        optimizer = torch.optim.Adam(nn.ParameterList(model.parameters()))
        models['base-model'], measures_base = train_clf(model, optimizer, meta,
                                            X_train, y_train, X_valid, y_valid)

        """ Train model without EMMA noisy train-set, eval on noisy valid-set """
        model = Model(d_input=np.sum(d_input)).float()
        model.load_state_dict(models['base-model'].state_dict())
        optimizer = torch.optim.Adam(nn.ParameterList(model.parameters()))
        models['model-without'], measures_without = train_clf(model, optimizer, meta,
                                            X_train, y_train, X_valid, y_valid)

        """ Train model with EMMA noisy train-set, eval on noisy valid-set """
        model = Model(d_input=np.sum(d_input)).float()
        model.load_state_dict(models['base-model'].state_dict())
        emma = EMMA(n_modes, list(autoencoders.values()), min_potentials).float()
        model = nn.ModuleList([emma, model])
        optimizer = torch.optim.Adam(nn.ParameterList(model.parameters()))
        models['model-with'], measures_with = train_clf(model, optimizer, meta,
                                            X_train, y_train, X_valid, y_valid,
                                            with_emma=True)

        """ Retrain autoencoders on concat normal signal train-set + valid-set """
        X_signal = torch.cat((X_signal_train, X_signal_valid), dim=0)
        for key, model in autoencoders.items():
            unfreeze(model)
        
        model = DenoisingAutoEncoder(d_input[0], n_hidden, noise_std_autoenc).float()
        model.load_state_dict(autoencoders['integrated-profile'].state_dict())
        optimizer = torch.optim.Adam(nn.ParameterList(model.parameters()))
        autoencoders['integrated-profile'], _ = train_autoencoder(model, optimizer,
                128, 10, X_signal_train[:,:4])
        min_potentials[0] = get_min_potential(X_signal[:,:4], model)
        torch.save((model.state_dict(), min_potentials[0]), "dumps/autoencoder-ip.pt")

        model = DenoisingAutoEncoder(d_input[1], n_hidden, noise_std_autoenc).float()
        model.load_state_dict(autoencoders['DM-SNR'].state_dict())
        optimizer = torch.optim.Adam(nn.ParameterList(model.parameters()))
        autoencoders['DM-SNR'], _ = train_autoencoder(model, optimizer, 128, 10,
                X_signal_train[:,4:])
        min_potentials[1] = get_min_potential(X_signal[:,4:], model)
        torch.save((model.state_dict(), min_potentials[1]), "dumps/autoencoder-dm-snr.pt")

        for key, model in autoencoders.items():
            freeze(model)

        """ Retrain base model on concat normal train-set + noisy valid-set """
        X = torch.cat((X_train, X_valid), dim=0)
        y = torch.cat((y_train, y_valid), dim=0)

        meta['max_epochs'] = int(meta['max_epochs']/2)
        model = Model(d_input=np.sum(d_input)).float()
        model.load_state_dict(models['base-model'].state_dict())
        optimizer = torch.optim.Adam(nn.ParameterList(model.parameters()))
        models['base-model'], _ = train_clf(model, optimizer, meta, X, y)
        torch.save((model.state_dict(), measures_base), "dumps/base-model.pt")

        """ Retrain model without EMMA on concat noisy train-set + noisy valid-set """
        model = Model(d_input=np.sum(d_input)).float()
        model.load_state_dict(models['model-without'].state_dict())
        optimizer = torch.optim.Adam(nn.ParameterList(model.parameters()))
        models['model-without'], _ = train_clf(model, optimizer, meta, X, y)
        torch.save((model.state_dict(), measures_without), "dumps/model-without.pt")

        """ Retrain model with EMMA on concat noisy train-set + noisy valid-set """
        model = Model(d_input=np.sum(d_input)).float()
        emma = EMMA(n_modes, list(autoencoders.values()), min_potentials).float()
        model = nn.ModuleList([emma, model])
        model.load_state_dict(models['model-with'].state_dict())
        optimizer = torch.optim.Adam(nn.ParameterList(model.parameters()))
        models['model-with'], _ = train_clf(model, optimizer, meta, X, y, with_emma=True)
        torch.save((model.state_dict(), measures_with), "dumps/model-with.pt")

    else:
        """ Load autoencoders """
        autoencoders['integrated-profile'] = DenoisingAutoEncoder(d_input[0], n_hidden, noise_std_autoenc).float()
        params, min_potentials[0] = torch.load("dumps/autoencoder-ip.pt")
        autoencoders['integrated-profile'].load_state_dict(params.state_dict())

        autoencoders['DM-SNR'] = DenoisingAutoEncoder(d_input[1], n_hidden, noise_std_autoenc).float()
        params, min_potentials[1] = torch.load("dumps/autoencoder-dm-snr.pt")
        autoencoders['DM-SNR'].load_state_dict(params.state_dict())

        """ Load base model """
        models['base-model'] = Model(d_input=np.sum(d_input)).float()
        params, measures_base = torch.load("dumps/base-model.pt")
        models['base-model'].load_state_dict(params.state_dict())

        """ Load model-without """
        models['model-without'] = Model(d_input=np.sum(d_input)).float()
        params, measures_without = torch.load("dumps/model-without.pt")
        models['model-without'].load_state_dict(params.state_dict())

        """ Load model-with """
        models['model-with'] = Model(d_input=np.sum(d_input)).float()
        emma = EMMA(n_modes, autoencoders.values(), min_potentials).float()
        model = nn.ModuleList([emma, model])
        params, measures_with = torch.load("dumps/model-with.pt")
        models['model-with'].load_state_dict(params.state_dict())
    
    # Clone noisy test-set with indicator
    # Eval the three models on noisy test-set
    
















