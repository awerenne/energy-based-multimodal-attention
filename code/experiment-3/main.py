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
def regul_loss(indicator, log_alphas):
    return (indicator * log_alphas).sum()


# ---------------
def indic2regul(indicator):
    regul_indic = torch.zeros(indicator.size()).float()
    regul_indic[indicator[:,0] == 1, 1] = -1
    regul_indic[indicator[:,1] == 1, 0] = -1
    return -regul_indic


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
        y_valid=None, with_emma=False, indicator=None):
    n_epochs = meta['max_epochs']
    if not with_emma:
        measures = torch.zeros(n_epochs, 4, requires_grad=False)
        for epoch in range(n_epochs):
            model_original.train()
            train_step_clf(X_train, y_train, model_original, optimizer, meta, train=True)
            if (not X_valid is None) and (not y_valid is None):
                model_original.eval()
                with torch.set_grad_enabled(False):
                    best_threshold, scores = train_step_clf(X_valid, y_valid, model_original,
                            optimizer, meta, train=False)
                    measures[epoch, 0] = best_threshold
                    measures[epoch, 1] = scores[0]
                    measures[epoch, 2] = scores[1]
                    measures[epoch, 3] = scores[2]
        if (not X_valid is None) and (not y_valid is None):
            return model_original, measures
        return model_original
    
    coldness = meta['coldness']
    lambdas = meta['lambda']
    best_model = None
    best_F1 = -float("Inf")
    measures = torch.zeros(len(coldness), len(lambdas), n_epochs, 5, requires_grad=False)
    for i, tau in enumerate(coldness):
        for j, lambda_ in enumerate(lambdas):
            model = copy_emma_model(model_original)
            model[0].set_coldness(tau)
            optimizer = torch.optim.Adam(nn.ParameterList(model.parameters()))
            for epoch in range(n_epochs):
                model.train()
                train_step_clf(X_train, y_train, model, optimizer, meta, True, True, lambda_, indicator=indicator)
                if (not X_valid is None) and (not y_valid is None):
                    model.eval()
                    with torch.set_grad_enabled(False):
                        best_threshold, scores = train_step_clf(X_valid, y_valid, model,
                                optimizer, meta, train=False, with_emma=True,
                                lambda_=lambda_, indicator=indicator)
                        measures[i, j, epoch, 0] = best_threshold
                        measures[i, j, epoch, 1] = scores[0]
                        measures[i, j, epoch, 2] = scores[1]
                        measures[i, j, epoch, 3] = scores[2]
                        measures[i, j, epoch, 4] = model[0].capacity.data
                        if scores[0] > best_F1:
                            best_model = model
                            best_F1 = scores[0]

    if (not X_valid is None) and (not y_valid is None):
        return best_model, measures
    return model


# ---------------
def train_step_clf(X, y, model, optimizer, meta, train=True, with_emma=False,
        lambda_=None, indicator=None):
    n_steps = 0
    thresholds = np.linspace(0.1,0.9,9)
    scores = np.zeros((11, 3))
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
                loss = meta['criterion'](yhat, y[idx].unsqueeze(-1)) - \
                            lambda_ * regul_loss(indicator[idx], log_alphas)
            loss.backward()
            optimizer.step()
            if with_emma:
                model[0].apply(meta['clipper'])
        else:
            for j, threshold in enumerate(thresholds):
                f1, precision, recall = compute_F1(y[idx], yhat, threshold)
                scores[j, 0] += f1
                scores[j, 1] += precision
                scores[j, 2] += recall
        n_steps += 1
    if not train:
        scores /= n_steps
        idx = np.argmax(scores[:,0])
        return thresholds[idx], scores[idx,:]


# ---------------
def evaluation(X, y, model, threshold, with_emma):
    f1, precision, recall = 0, 0, 0
    n_steps = 0
    batch_size = 32
    indices = np.arange(X.size(0))
    for i in range(0, len(X)-batch_size, batch_size):
        idx = indices[i:i+batch_size]
        batch = X[idx].view(batch_size, -1)
        if not with_emma:
            yhat = model(batch)
        else:
            modes = [batch[:,:4], batch[:,4:]]
            modes, log_alphas = model[0](modes, False)
            new_batch = torch.zeros(batch.size()).float()
            new_batch[:, :4], new_batch[:, 4:]  = modes[0], modes[1]
            yhat = model[1](new_batch)  # N x D
        f1_, precision_, recall_ = compute_F1(y[idx], yhat, threshold)
        f1 += f1_
        precision += precision_
        recall += recall_
        n_steps += 1
    return (f1/n_steps).data.numpy(), (precision/n_steps).data.numpy(), (recall/n_steps).data.numpy()


# ---------------
def model_evaluation(X, y, indic, models, thresholds):
    for name, model in models.items():
        print(name + ": ")
        if name == "model-with":
            with_emma = True
        else: 
            with_emma = False
        threshold = thresholds[name]

        """ Average """
        F1, precision, recall = evaluation(X, y, model, threshold, with_emma)
        print("  Average: " + str(F1) + " - " + str(precision) + " - " + str(recall))
        
        idx = (indic[:,0] == 0) * (indic[:,1] == 0)
        F1, precision, recall = evaluation(X[idx], y[idx], model, threshold, with_emma)
        print("  Normal: " + str(F1) + " - " + str(precision) + " - " + str(recall))
        
        idx = (indic[:,0] == 1) * (indic[:,1] == 0)
        F1, precision, recall = evaluation(X[idx], y[idx], model, threshold, with_emma)
        print("  Noisy-ip: " + str(F1) + " - " + str(precision) + " - " + str(recall))
        
        idx = (indic[:,0] == 0) * (indic[:,1] == 1)
        F1, precision, recall = evaluation(X[idx], y[idx], model, threshold, with_emma)
        print("  Noisy-dm: " + str(F1) + " - " + str(precision) + " - " + str(recall))


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


# ---------------
if __name__ == "__main__":  
    """ Parameters experiment """
    retrain = False
    n_modes = 2
    d_input = [4, 4]
    n_hidden = 12  # Number of hidden units in autoencoders
    noise_std_autoenc = 0.01
    noise_std_data = 1.2
    coldness = [0, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e2, 1e3, 1e4]
    lambda_ = np.linspace(0, 2, 5)

    meta = {}
    meta['criterion'] = nn.BCELoss()
    meta['clipper'] = WeightClipper()
    meta['coldness'] = coldness
    meta['lambda'] = lambda_
    meta['max_epochs'] = 25
    meta['batch_size'] = 128

    """ Data """
    train_set, valid_set, test_set = get_pulsar_data("../data/pulsar.csv")
    X_train, y_train = train_set
    X_train_noisy, y_train_noisy, indic_train = apply_corruption(X_train, y_train,
                                                    noise_std_data)
    X_valid, y_valid = valid_set
    X_valid_noisy, y_valid_noisy, indic_valid = apply_corruption(X_valid, y_valid,
                                                    noise_std_data)
    X_test, y_test = test_set
    X_test_noisy, y_test_noisy, indic_test = apply_corruption(X_test, y_test,
                                                    noise_std_data)

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
        model, _ = train_autoencoder(model, optimizer,
                64, 30, X_signal_train[:,:4], X_signal_valid[:,:4])
        autoencoders['integrated-profile'] = model
        min_potentials[0] = get_min_potential(X_signal_train[:,:4], model)

        model = DenoisingAutoEncoder(d_input[1], n_hidden, noise_std_autoenc).float()
        optimizer = torch.optim.Adam(nn.ParameterList(model.parameters()))
        model, _ = train_autoencoder(model, optimizer, 64, 30,
                X_signal_train[:,4:], X_signal_valid[:,4:])
        autoencoders['DM-SNR'] = model
        min_potentials[1] = get_min_potential(X_signal_train[:,4:], model)
        min_potentials = torch.tensor(min_potentials).float()

        """ Freeze autoencoders """
        for key, model in autoencoders.items():
            freeze(model)

        """ Train base model on normal train-set, eval on noisy valid-set """
        model = Model(d_input=np.sum(d_input)).float()
        optimizer = torch.optim.Adam(nn.ParameterList(model.parameters()))
        model, measures_base = train_clf(model, optimizer, meta,
                                            X_train, y_train, X_valid_noisy, y_valid_noisy)
        models['base-model'] = model

        """ Train model without EMMA noisy train-set, eval on noisy valid-set """
        model = Model(d_input=np.sum(d_input)).float()
        model.load_state_dict(models['base-model'].state_dict())
        optimizer = torch.optim.Adam(nn.ParameterList(model.parameters()))
        model, measures_without = train_clf(model, optimizer, meta,
                                            X_train_noisy, y_train_noisy, X_valid_noisy, y_valid_noisy)
        models['model-without'] = model


        """ Train model with EMMA noisy train-set, eval on noisy valid-set """
        regul_indicator = indic2regul(indic_train)
        model = Model(d_input=np.sum(d_input)).float()
        model.load_state_dict(models['base-model'].state_dict())
        emma = EMMA(n_modes, list(autoencoders.values()), min_potentials).float()
        model = nn.ModuleList([emma, model])
        optimizer = torch.optim.Adam(nn.ParameterList(model.parameters()))
        model, measures_with = train_clf(model, optimizer, meta,
                                            X_train_noisy, y_train_noisy, X_valid_noisy, y_valid_noisy,
                                            with_emma=True, indicator=regul_indicator)
        models['model-with'] = model

        """ Retrain autoencoders on concat normal signal train-set + valid-set """
        X_signal = torch.cat((X_signal_train, X_signal_valid), dim=0)
        for key, model in autoencoders.items():
            unfreeze(model)
        
        model = DenoisingAutoEncoder(d_input[0], n_hidden, noise_std_autoenc).float()
        model.load_state_dict(autoencoders['integrated-profile'].state_dict())
        optimizer = torch.optim.Adam(nn.ParameterList(model.parameters()))
        model, _ = train_autoencoder(model, optimizer,
                128, 10, X_signal_train[:,:4])
        min_potentials[0] = get_min_potential(X_signal[:,:4], model)
        autoencoders['integrated-profile'] = model
        torch.save((model.state_dict(), min_potentials[0]), "dumps/autoencoder-ip.pt")

        model = DenoisingAutoEncoder(d_input[1], n_hidden, noise_std_autoenc).float()
        model.load_state_dict(autoencoders['DM-SNR'].state_dict())
        optimizer = torch.optim.Adam(nn.ParameterList(model.parameters()))
        model, _ = train_autoencoder(model, optimizer, 128, 10,
                X_signal_train[:,4:])
        autoencoders['DM-SNR'] = model
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
        model = train_clf(model, optimizer, meta, X, y)
        models['base-model'] = model
        torch.save((model.state_dict(), measures_base), "dumps/base-model.pt")

        """ Retrain model without EMMA on concat noisy train-set + noisy valid-set """
        X = torch.cat((X_train_noisy, X_valid_noisy), dim=0)
        y = torch.cat((y_train_noisy, y_valid_noisy), dim=0)

        model = Model(d_input=np.sum(d_input)).float()
        model.load_state_dict(models['model-without'].state_dict())
        optimizer = torch.optim.Adam(nn.ParameterList(model.parameters()))
        model = train_clf(model, optimizer, meta, X, y)
        models['model-without'] = model
        torch.save((model.state_dict(), measures_without), "dumps/model-without.pt")

        """ Retrain model with EMMA on concat noisy train-set + noisy valid-set """
        regul_indicator = torch.cat((indic2regul(indic_train), indic2regul(indic_valid)), dim=0)
        model = Model(d_input=np.sum(d_input)).float()
        emma = EMMA(n_modes, list(autoencoders.values()), min_potentials).float()
        model = nn.ModuleList([emma, model])
        model.load_state_dict(models['model-with'].state_dict())
        optimizer = torch.optim.Adam(nn.ParameterList(model.parameters()))
        m_ = measures_with.data.numpy()[:,:,-1,1]  # last epoch
        idx = np.unravel_index(np.argmax(m_, axis=None), m_.shape)
        meta['coldness'] = [coldness[idx[0]]]
        print(meta['coldness'])
        meta['lambda'] = [lambda_[idx[1]]]
        print(meta['lambda'])
        model = train_clf(model, optimizer, meta, X, y,
                                    with_emma=True, indicator=regul_indicator)
        models['model-with'] = model
        torch.save((model.state_dict(), model[0].get_coldness(), measures_with), "dumps/model-with.pt")
        print(model[0].get_coldness())

        """ Save noisy test-set (with indicator) """
        torch.save((X_test_noisy, y_test_noisy, indic_test), "dumps/test-set.pt")

    else:
        """ Load autoencoders """
        autoencoders['integrated-profile'] = DenoisingAutoEncoder(d_input[0], n_hidden, noise_std_autoenc).float()
        params, min_potentials[0] = torch.load("dumps/autoencoder-ip.pt")
        autoencoders['integrated-profile'].load_state_dict(params)

        autoencoders['DM-SNR'] = DenoisingAutoEncoder(d_input[1], n_hidden, noise_std_autoenc).float()
        params, min_potentials[1] = torch.load("dumps/autoencoder-dm-snr.pt")
        autoencoders['DM-SNR'].load_state_dict(params)
        min_potentials = torch.tensor(min_potentials).float()

        """ Load base model """
        models['base-model'] = Model(d_input=np.sum(d_input)).float()
        params, measures_base = torch.load("dumps/base-model.pt")
        models['base-model'].load_state_dict(params)

        """ Load model-without """
        models['model-without'] = Model(d_input=np.sum(d_input)).float()
        params, measures_without = torch.load("dumps/model-without.pt")
        models['model-without'].load_state_dict(params)

        """ Load model-with """
        model = Model(d_input=np.sum(d_input)).float()
        emma = EMMA(n_modes, list(autoencoders.values()), min_potentials).float()
        models['model-with'] = nn.ModuleList([emma, model])
        params, tau, measures_with = torch.load("dumps/model-with.pt")
        print(tau)
        models['model-with'].load_state_dict(params)
        models['model-with'][0].set_coldness(tau)

        """ Load noisy test-set (with indicator) """
        X_test_noisy, y_test_noisy, indic_test = torch.load("dumps/test-set.pt")
    
    # Clone noisy test-set with indicator
    # Eval the three models on noisy test-set
    # best_threshold = np.linspace(0.1,0.9,9)[-1,0]
    for model in models.values():
        model.eval()

    thresholds = {}
    thresholds['base-model'] = measures_base[-1,0]
    thresholds['model-without'] = measures_without[-1,0]
    m_ = measures_with.data.numpy()[:,:,-1,1]  # last epoch
    idx = np.unravel_index(np.argmax(m_, axis=None), m_.shape)
    thresholds['model-with'] = measures_with[idx[0], idx[1], -1, 0]
    print()
    print(thresholds)
    print(models['model-with'][0].get_coldness())
    print()
    model_evaluation(X_test_noisy, y_test_noisy, indic_test, models, thresholds)

    # plot_6(measures_with)
    # plot_7(measures_with)
    # plot_2(X_test_noisy, indic_test, models['model-with'])
















