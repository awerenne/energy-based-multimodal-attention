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
from itertools import combinations


# ---------------
class MLP(nn.Module):
    """
        ...
    """

    def __init__(self, d_input):
        super().__init__()
        # self.linear1 = nn.Linear(d_input, int(d_input/2))
        # self.linear2 = nn.Linear(int(d_input/2), 1)
        self.linear1 = nn.Linear(d_input, 8)
        self.linear2 = nn.Linear(8, 4)
        self.linear3 = nn.Linear(4, 1)
        self.threshold = 0.5
        self.d_input = d_input

    def set_threshold(self, threshold):
        self.threshold = threshold

    def get_threshold(self):
        return self.threshold

    def get_dim_input(self):
        return self.d_input

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        x = torch.relu(x)
        x = self.linear3(x)
        return torch.sigmoid(x), x


# ---------------
def copy_emma(old_model):
    mlp = MLP(d_input=old_model[1].get_dim_input()).float()
    emma = EMMA(old_model[0].n_modes, old_model[0].autoencoders, old_model[0].min_potentials).float()
    emma.set_coldness(old_model[0].get_coldness())
    model = nn.ModuleList([emma, mlp])
    model.load_state_dict(old_model.state_dict())
    return model


# ---------------
def copy_mlp(old_model):
    mlp = MLP(d_input=old_model.get_dim_input()).float()
    mlp.load_state_dict(old_model.state_dict())
    return mlp


# ---------------
def regul_loss(indicator, log_alphas):
    loss_indic = torch.zeros(indicator.size()).float()
    loss_indic[indicator[:,0] == 1, 1] = -1
    loss_indic[indicator[:,1] == 1, 0] = -1
    loss_indic *= -1
    return (loss_indic * log_alphas).sum()


# ---------------
def train_autoencoder(model, optimizer, batch_size, n_epochs, X_train, X_valid=None):
    valid_curve = []
    for epoch in range(n_epochs):
        if not X_valid is None:
            model.eval()
            with torch.set_grad_enabled(False):
                loss = train_step_autoencoder(X_valid, model, optimizer, batch_size, train=False)
            valid_curve.append(loss)
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
def train_clf(models, name, meta, X_train, y_train, indicator_train,
                X_valid, y_valid, indic_valid):
    X = torch.cat((X_train, X_valid), dim=0)
    y = torch.cat((y_train, y_valid), dim=0)
    indic = torch.cat((indic_train, indic_valid), dim=0)

    """ Train models without emma """
    if not name == 'model-with':
        n_epochs = meta['max_epochs']
        best_loss = float("Inf")
        best_model = None
        best_epoch = 0
        state_optim = None
        train_curve, valid_curve = np.zeros(n_epochs+1), np.zeros(n_epochs+1)
        model = models[name]
        optimizer = torch.optim.Adam(nn.ParameterList(model.parameters()))
        model.eval()
        with torch.set_grad_enabled(False):
            valid_curve[0] = train_step(model, name, optimizer, meta['batch_size'], X_train, y_train, valid=True)     
        for epoch in range(1, n_epochs+1):
            model.train()
            train_curve[epoch] = train_step(model, name, optimizer, meta['batch_size'], X_train, y_train)
            model.eval()
            with torch.set_grad_enabled(False):
                valid_curve[epoch] = train_step(model, name, optimizer, meta['batch_size'], X_train, y_train, valid=True)
            if valid_curve[epoch] < best_loss:
                best_loss = valid_curve[epoch]
                best_model = copy_mlp(model)
                best_epoch = epoch
                state_optim = optimizer.state_dict()
        optimizer = torch.optim.Adam(nn.ParameterList(best_model.parameters()))
        optimizer.load_state_dict(state_optim)
        threshold = optimal_threshold(best_model, name, X_valid, y_valid)
        best_model.set_threshold(threshold)
        n_epochs = int(0.5 * y_valid.size(0)/y_train.size(0) * best_epoch)
        best_model.train()  
        for epoch in range(best_epoch):
            train_step(best_model, name, optimizer, meta['batch_size'], X, y)
        models[name] = best_model
        return (train_curve, valid_curve)
        
    """ Train models with emma """
    for tau in meta['coldness']:
        for lambda_regul in meta['lambda_regul']:
            for lambda_capacity in meta['lambda_capacity']:
                print("Coldness: " + str(tau) + \
                        ", lambda regul: " + str(lambda_regul) + \
                        ", lambda capacity: " + str(lambda_capacity))
                n_epochs = meta['max_epochs']
                best_loss = float("Inf")
                best_model = None
                best_epoch = 0
                state_optim = None
                train_curve, valid_curve = np.zeros(n_epochs+1), np.zeros(n_epochs+1)
                model = copy_emma(models[name][(-1,-1,-1)])
                model[0].set_coldness(tau)
                optimizer = torch.optim.Adam(nn.ParameterList(model.parameters()))
                model.eval()
                with torch.set_grad_enabled(False):
                    valid_curve[0] = train_step(model, name, optimizer, meta['batch_size'], X_train, y_train, indic_train, lambda_regul, lambda_capacity, valid=True)
                for epoch in range(1, n_epochs+1):
                    model.train()
                    train_curve[epoch] = train_step(model, name, optimizer, meta['batch_size'], X_train, y_train, indic_train, lambda_regul, lambda_capacity)
                    model.eval()
                    with torch.set_grad_enabled(False):
                        valid_curve[epoch] = train_step(model, name, optimizer, meta['batch_size'], X_train, y_train, indic_train, lambda_regul, lambda_capacity, valid=True)
                    if valid_curve[epoch] < best_loss:
                        best_loss = valid_curve[epoch]
                        best_model = copy_emma(model)
                        state_optim = optimizer.state_dict()
                optimizer = torch.optim.Adam(nn.ParameterList(best_model.parameters()))
                optimizer.load_state_dict(state_optim)
                threshold = optimal_threshold(best_model, name, X_valid, y_valid)
                best_model[1].set_threshold(threshold) 
                n_epochs = int(0.5 * y_valid.size(0)/y_train.size(0) * best_epoch)
                best_model.train()  
                for epoch in range(n_epochs):
                    train_step(best_model, name, optimizer, meta['batch_size'], X, y)
                models[name][tau, lambda_regul, lambda_capacity] = best_model
    del models[name][(-1,-1,-1)]
    return (train_curve, valid_curve)


# ---------------
def train_step(model, name, optimizer, batch_size, X, y, indic=None, lambda_regul=None,
                lambda_capacity=None, valid=False):
    n_steps, sum_loss = 0, 0
    # criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.]))
    criterion = torch.nn.BCEWithLogitsLoss()
    # criterion = torch.nn.BCELoss()
    indices = np.arange(X.size(0))
    np.random.shuffle(indices)
    for i in range(0, len(X)-batch_size, batch_size):
        optimizer.zero_grad()
        idx = indices[i:i+batch_size]
        batch = X[idx].view(batch_size, -1)
        if not name == 'model-with':
            yhat, logits = model(batch)
            loss = criterion(logits, y[idx].unsqueeze(-1))
            # loss = criterion(yhat, y[idx].unsqueeze(-1))
        else:
            modes = [batch[:,:4], batch[:,4:]]
            modes, log_alphas = model[0](modes)
            new_batch = torch.zeros(batch.size()).float()
            new_batch[:, :4], new_batch[:, 4:]  = modes[0], modes[1]
            yhat, logits = model[1](new_batch)  
            loss = criterion(logits, y[idx].unsqueeze(-1))
            # loss = criterion(yhat, y[idx].unsqueeze(-1))
            loss -= lambda_regul * regul_loss(indic[idx], log_alphas)
            loss += (lambda_capacity * model[0].get_gain())[0]
        sum_loss += loss.item()
        if valid == False:
            loss.backward()
            optimizer.step()
        if name == 'model-with': model[0].apply(meta['clipper'])
        n_steps += 1
    return sum_loss/n_steps


# ---------------
def optimal_threshold(model, name, X, y):
    thresholds = np.linspace(0,1,21)
    best_dist = float("Inf")
    best_threshold = 0.5
    if not name == 'model-with':
        yhat, _ = model(X)
    else:
        modes = [X[:,:4], X[:,4:]]
        modes, log_alphas = model[0](modes)
        new_X = torch.zeros(X.size()).float()
        new_X[:, :4], new_X[:, 4:]  = modes[0], modes[1]
        yhat, _ = model[1](new_X)  
    for threshold in thresholds:
        _, _, TPR, specificity = compute_score(y, yhat, threshold)
        FPR = 1 - specificity
        dist = np.sqrt((FPR-0)**2 + (1-TPR)**2)
        if dist < best_dist:
            best_dist = dist
            best_threshold = threshold
    return best_threshold


# ---------------
def predictions(model, name, X):
    if not name == 'model-with':
        yhat, _ = model(X)
        classes = (yhat >= model.get_threshold()).clone().squeeze() 
    else:
        modes = [X[:,:4], X[:,4:]]
        modes, log_alphas = model[0](modes)
        new_X = torch.zeros(X.size()).float()
        new_X[:, :4], new_X[:, 4:] = modes[0], modes[1]
        yhat, _ = model[1](new_X)  # N x D
        classes = (yhat >= model[1].get_threshold()).clone().squeeze() 
    return yhat, classes


# ---------------
def evaluation(model, name, X, y):
    if not name == 'model-with':
        yhat, _ = model(X)
        score = compute_score(y, yhat, model.get_threshold())
    else:
        modes = [X[:,:4], X[:,4:]]
        modes, log_alphas = model[0](modes)
        new_X = torch.zeros(X.size()).float()
        new_X[:, :4], new_X[:, 4:] = modes[0], modes[1]
        yhat, _ = model[1](new_X)  # N x D
        score = compute_score(y, yhat, model[1].get_threshold())
    return score[0], score[1], score[2], score[3] 


# ---------------
def model_evaluation(model, name, X, y, indic):
        idx = (indic[:,0] == 0) * (indic[:,1] == 0)
        F1, precision, recall, specificity = evaluation(model, name, X[idx], y[idx])
        print("    Normal: " + str("{:.4f}".format(F1)) + " - " + str("{:.4f}".format(precision)) + " - " + str("{:.4f}".format(recall)) + " - " + str("{:.4f}".format(specificity)))
        
        idx = (indic[:,0] == 1) * (indic[:,1] == 0)
        F1, precision, recall, specificity = evaluation(model, name, X[idx], y[idx])
        print("  Noisy-ip: " + str("{:.4f}".format(F1)) + " - " + str("{:.4f}".format(precision)) + " - " + str("{:.4f}".format(recall)) + " - " + str("{:.4f}".format(specificity)))
        
        idx = (indic[:,0] == 0) * (indic[:,1] == 1)
        F1, precision, recall, specificity = evaluation(model, name, X[idx], y[idx])
        print("  Noisy-dm: " + str("{:.4f}".format(F1)) + " - " + str("{:.4f}".format(precision)) + " - " + str("{:.4f}".format(recall)) + " - " + str("{:.4f}".format(specificity)))


# ---------------
def compute_score(y, yhat, threshold):
    yhat_ = (yhat >= threshold).clone().squeeze()
    y_ = y.byte().clone()
    TP = true_positives(y_, yhat_)
    FP = false_positives(y_, yhat_)
    TN = true_negatives(y_, yhat_)
    FN = false_negatives(y_, yhat_)
    if TP + FP == 0: precision = 0
    else: precision = TP.float() / (TP+FP).float()
    if TP + FN == 0: recall = 0
    else: recall = TP.float() / (TP+FN).float()
    if TN + FP == 0: specificity = 0
    else: specificity = TN.float() / (TN+FP).float()
    if precision + recall == 0: f1_score = 0
    else: f1_score = (2 * precision * recall) / (precision + recall)
    return np.asarray([f1_score, precision, recall, specificity])


# ---------------
def true_positives(y, yhat):
    return ((y == 1) * (yhat == 1)).sum()


# ---------------
def false_positives(y, yhat):
    return ((y == 0) * (yhat == 1)).sum()
    

# ---------------
def true_negatives(y, yhat):
    return ((y == 0) * (yhat == 0)).sum()


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
    retrain = True
    n_modes = 2
    d_input = [4, 4]
    n_hidden = 12  # Number of hidden units in autoencoders
    noise_std_autoenc = 0.01
    noise_std_data = 10
    # coldness = [0, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e2, 1e3, 1e4]
    # lambda_ = np.linspace(0, 2, 5)
    coldness = [1e-2, 1e-1, 1]
    lambda_regul = np.linspace(0, 2, 3)
    lambda_capacity = np.linspace(0, 2, 3)

    meta = {}
    meta['clipper'] = WeightClipper()
    meta['coldness'] = coldness
    meta['lambda_regul'] = lambda_regul
    meta['lambda_capacity'] = lambda_capacity
    meta['max_epochs'] = 20
    meta['batch_size'] = 32

    """ Data """
    train_set, valid_set, test_set = get_pulsar_data("../data/pulsar.csv")
    X_train, y_train = train_set
    X_train_noisy, y_train_noisy, indic_train = apply_corruption(X_train, y_train,
                                                    noise_std_data)
    X_valid, y_valid = valid_set
    X_valid_noisy, y_valid_noisy, indic_valid = apply_corruption(X_valid, y_valid,
                                                    noise_std_data)
    X_test, y_test = test_set
    X_test_noisy, y_test_noisy, indic_test = apply_corruption_test(X_test, y_test,
                                                    noise_std_data)

    """ EDA """
    # table_occur = np.zeros((9,2))
    # table_occur[0,0] = int(((y_train_noisy == 1) * (indic_train[:,0] == 0) * (indic_train[:,1] == 0)).sum().data.numpy())
    # table_occur[0,1] = int(((y_train_noisy == 0) * (indic_train[:,0] == 0) * (indic_train[:,1] == 0)).sum().data.numpy())
    # table_occur[1,0] = int(((y_train_noisy == 1) * (indic_train[:,0] == 1) * (indic_train[:,1] == 0)).sum().data.numpy())
    # table_occur[1,1] = int(((y_train_noisy == 0) * (indic_train[:,0] == 1) * (indic_train[:,1] == 0)).sum().data.numpy())
    # table_occur[2,0] = int(((y_train_noisy == 1) * (indic_train[:,0] == 0) * (indic_train[:,1] == 1)).sum().data.numpy())
    # table_occur[2,1] = int(((y_train_noisy == 0) * (indic_train[:,0] == 0) * (indic_train[:,1] == 1)).sum().data.numpy())
    # table_occur[3,0] = int(((y_valid_noisy == 1) * (indic_valid[:,0] == 0) * (indic_valid[:,1] == 0)).sum().data.numpy())
    # table_occur[3,1] = int(((y_valid_noisy == 0) * (indic_valid[:,0] == 0) * (indic_valid[:,1] == 0)).sum().data.numpy())
    # table_occur[4,0] = int(((y_valid_noisy == 1) * (indic_valid[:,0] == 1) * (indic_valid[:,1] == 0)).sum().data.numpy())
    # table_occur[4,1] = int(((y_valid_noisy == 0) * (indic_valid[:,0] == 1) * (indic_valid[:,1] == 0)).sum().data.numpy())
    # table_occur[5,0] = int(((y_valid_noisy == 1) * (indic_valid[:,0] == 0) * (indic_valid[:,1] == 1)).sum().data.numpy())
    # table_occur[5,1] = int(((y_valid_noisy == 0) * (indic_valid[:,0] == 0) * (indic_valid[:,1] == 1)).sum().data.numpy())
    # table_occur[6,0] = int(((y_test_noisy == 1) * (indic_test[:,0] == 0) * (indic_test[:,1] == 0)).sum().data.numpy())
    # table_occur[6,1] = int(((y_test_noisy == 0) * (indic_test[:,0] == 0) * (indic_test[:,1] == 0)).sum().data.numpy())
    # table_occur[7,0] = int(((y_test_noisy == 1) * (indic_test[:,0] == 1) * (indic_test[:,1] == 0)).sum().data.numpy())
    # table_occur[7,1] = int(((y_test_noisy == 0) * (indic_test[:,0] == 1) * (indic_test[:,1] == 0)).sum().data.numpy())
    # table_occur[8,0] = int(((y_test_noisy == 1) * (indic_test[:,0] == 0) * (indic_test[:,1] == 1)).sum().data.numpy())
    # table_occur[8,1] = int(((y_test_noisy == 0) * (indic_test[:,0] == 0) * (indic_test[:,1] == 1)).sum().data.numpy())
    # print(table_occur)

    """ Training """
    autoencoders = {'IP': None, 'DM-SNR': None}
    min_potentials = [None, None]
    models = {'base-model': None, 'model-without': None, 'model-with': {(-1,-1,-1): None}}
    curves = {'base-model': None, 'model-without': None, 'model-with': {(-1,-1,-1): None}}
    if retrain:
        """ Train autoencoders on normal signal train-set """
        # X_signal_train = signal_only(X_train, y_train)
        # X_signal_valid = signal_only(X_valid, y_valid)
        # model = DenoisingAutoEncoder(d_input[0], n_hidden, noise_std_autoenc).float()
        # optimizer = torch.optim.Adam(nn.ParameterList(model.parameters()))
        # model, _ = train_autoencoder(model, optimizer, meta['batch_size'], 30, X_signal_train[:,:4], X_signal_valid[:,:4])
        # autoencoders['IP'] = model
        # min_potentials[0] = get_min_potential(X_signal_train[:,:4], model)

        # model = DenoisingAutoEncoder(d_input[1], n_hidden, noise_std_autoenc).float()
        # optimizer = torch.optim.Adam(nn.ParameterList(model.parameters()))
        # model, _ = train_autoencoder(model, optimizer, meta['batch_size'], 30, X_signal_train[:,4:], X_signal_valid[:,4:])
        # autoencoders['DM-SNR'] = model
        # min_potentials[1] = get_min_potential(X_signal_train[:,4:], model)
        # min_potentials = torch.tensor(min_potentials).float()

        # """ Freeze autoencoders """
        # for key, autoencoder in autoencoders.items():
        #     freeze(autoencoder)

        """ Train base model on normal train-set, eval on valid-set """
        model = MLP(d_input=np.sum(d_input)).float()
        models['base-model'] = model
        train_clf(models, 'base-model', meta, X_train, y_train, indic_train, 
            X_valid, y_valid, indic_valid)

        """ Train model without EMMA noisy train-set, eval on noisy valid-set """
        # mlp = copy_mlp(models['base-model'])
        # models['model-without'] = mlp
        # train_clf(models, 'model-without', meta, X_train_noisy, y_train_noisy, indic_train, 
        #     X_valid_noisy, y_valid_noisy, indic_valid)

        # """ Train model with EMMA noisy train-set, eval on noisy valid-set """
        # mlp = copy_mlp(models['base-model'])
        # emma = EMMA(n_modes, list(autoencoders.values()), min_potentials).float()
        # model = nn.ModuleList([emma, mlp])
        # models['model-with'][(-1,-1,-1)] = model
        # train_clf(models, 'model-with', meta, X_train_noisy, y_train_noisy, indic_train, 
        #     X_valid_noisy, y_valid_noisy, indic_valid)

        # """ Save """
        # torch.save((models), "dumps/models")
        # torch.save((X_test_noisy, y_test_noisy, indic_test), "dumps/test-set.pt")

    else:
        """ Load models """
        models = torch.load("dumps/models")

        # """ Load noisy test-set (with indicator) """
        X_test_noisy, y_test_noisy, indic_test = torch.load("dumps/test-set.pt")
    

    """ Evaluation """
    print("F1-score   Precision   Recall   specificity")
    for name, model in models.items():
        print()
        print(name + ": ")
        if name == 'model-with':
            for key, val in model.items():
                print(key)
                val.eval()
                model_evaluation(val, name, X_test_noisy, y_test_noisy, indic_test)
        else:
            model.eval()
            model_evaluation(model, name, X_test_noisy, y_test_noisy, indic_test)

    """ ... """
    # ranking = [models['model-with'][(0.01,1,0)], models['model-with'][(1,0,0)]]
    # yhat, classes = predictions(models['base-model'], 'base-model', X_test_noisy) 
    # plot_confusion_matrix(classes.data.numpy(), y_test_noisy.data.numpy())
    # yhat, classes = predictions(models['model-without'], 'model-without', X_test_noisy) 
    # plot_confusion_matrix(classes.data.numpy(), y_test_noisy.data.numpy())
    # yhat, classes = predictions(ranking[0], 'model-with', X_test_noisy) 
    # plot_confusion_matrix(classes.data.numpy(), y_test_noisy.data.numpy())


    # table_occur = np.zeros((6,2))
    # table_occur[0,0] = int(((y_test_noisy == 1) * (classes == 1) * (indic_test[:,0] == 0) * (indic_test[:,1] == 0)).sum().data.numpy())
    # table_occur[0,1] = int(((y_test_noisy == 1) * (classes == 0) * (indic_test[:,0] == 0) * (indic_test[:,1] == 0)).sum().data.numpy())
    
    # table_occur[1,0] = int(((y_test_noisy == 1) * (classes == 1) * (indic_test[:,0] == 1) * (indic_test[:,1] == 0)).sum().data.numpy())
    # table_occur[1,1] = int(((y_test_noisy == 1) * (classes == 0) * (indic_test[:,0] == 1) * (indic_test[:,1] == 0)).sum().data.numpy())

    # table_occur[2,0] = int(((y_test_noisy == 1) * (classes == 1) * (indic_test[:,0] == 0) * (indic_test[:,1] == 1)).sum().data.numpy())
    # table_occur[2,1] = int(((y_test_noisy == 1) * (classes == 0) * (indic_test[:,0] == 0) * (indic_test[:,1] == 1)).sum().data.numpy())

    # table_occur[3,0] = int(((y_test_noisy == 0) * (classes == 0) * (indic_test[:,0] == 0) * (indic_test[:,1] == 0)).sum().data.numpy())
    # table_occur[3,1] = int(((y_test_noisy == 0) * (classes == 1) * (indic_test[:,0] == 0) * (indic_test[:,1] == 0)).sum().data.numpy())

    # table_occur[4,0] = int(((y_test_noisy == 0) * (classes == 0) * (indic_test[:,0] == 1) * (indic_test[:,1] == 0)).sum().data.numpy())
    # table_occur[4,1] = int(((y_test_noisy == 0) * (classes == 1) * (indic_test[:,0] == 1) * (indic_test[:,1] == 0)).sum().data.numpy())

    # table_occur[5,0] = int(((y_test_noisy == 0) * (classes == 0) * (indic_test[:,0] == 0) * (indic_test[:,1] == 1)).sum().data.numpy())
    # table_occur[5,1] = int(((y_test_noisy == 0) * (classes == 1) * (indic_test[:,0] == 0) * (indic_test[:,1] == 1)).sum().data.numpy())

    # print(table_occur/np.sum(table_occur, axis=1, keepdims=True))

    # for name, param in ranking[0].named_parameters():
    #     print (name, param.data)
    
    # plot_2(ranking[0], X_test_noisy, indic_test)

    # coldness = [1e-2, 1e-1, 1]
    # f1 = []
    # rec = []
    # for tau in coldness:
    #     model = models['model-with'][(tau,1,0)]
    #     F1, _, recall, _ = evaluation(model, 'model-with', X_test_noisy, y_test_noisy)
    #     f1.append(F1)
    #     rec.append(recall)
    # plt.plot(coldness, f1)
    # plt.plot(coldness, rec)
    # plt.show()

    # coldness = [1e-2, 1e-1, 1]
    # c = []
    # for tau in coldness:
    #     model = models['model-with'][(tau,1,0)]
    #     c.append(model[0].capacity)
    # plt.plot(np.log(coldness), c)
    # plt.show()

    # f1_with = []
    # f1_without = []
    # for noise in np.linspace(0,80,50):
    #     X, y = torch.zeros(X_test.size()).float(), torch.zeros(y_test.size()).float()
    #     for i in range(4):
    #         X[:int(X.size(0)/2), i] = torch.tensor(white_noise(X_test[:int(X.size(0)/2), i].data.numpy(), noise)).float()
    #         X[int(X.size(0)/2):, i+4] = torch.tensor(white_noise(X_test[int(X.size(0)/2):, i+4].data.numpy(), noise)).float()
    #     # F1, _, _, _ = evaluation(ranking[0], 'model-with', X, y_test)
    #     # f1_with.append(F1)
    #     F1, _, _, _ = evaluation(models['model-with'][(0.01,0,2)], 'model-with', X, y_test)
    #     f1_with.append(F1)
    #     F1, _, _, _ = evaluation(models['model-without'], 'model-without', X, y_test)
    #     f1_without.append(F1)
    # plt.plot(np.linspace(0,80,50), f1_with, label='with')
    # plt.plot(np.linspace(0,80,50), f1_without, label='without')
    # plt.legend()
    # plt.show()

    # X, y = torch.zeros(X_test.size()).float(), torch.zeros(y_test.size()).float()
    # for i in range(4):
    #     X[:int(X.size(0)/2), i] = torch.tensor(white_noise(X_test[:int(X.size(0)/2), i].data.numpy(), 5)).float()
    #     X[int(X.size(0)/2):, i+4] = torch.tensor(white_noise(X_test[int(X.size(0)/2):, i+4].data.numpy(), 5)).float()
    # res = np.zeros((3,3))
    # coldness = [1e-2, 1e-1, 1]
    # lambda_regul = np.linspace(0, 2, 3)
    # lambda_capacity = np.linspace(0, 2, 3)
    # for i, l_reg in enumerate(lambda_regul):
    #     for j, l_cap in enumerate(lambda_capacity):
    #         best_F1 = -float("Inf")
    #         for tau in coldness:
    #             F1, _, _, _ = evaluation(models['model-with'][(tau,l_reg,l_cap)], 'model-with', X, y_test)
    #             if F1 > best_F1:
    #                 best_F1 = F1
    #         res[i,j] = best_F1
    # sns.heatmap(res, annot=True, cmap="YlGnBu")
    # plt.show()

    # def mixup(X, y, indic, mixing):
    #     if mixing == 0:
    #         return X, y
    #     idx_signal = (y == 1) * (indic[:,0] == 0) * (indic[:,1] == 0)
    #     idx_signal = np.where(idx_signal.data.numpy())[0]
    #     idx_bckg = (y == 1) * (indic[:,0] == 0) * (indic[:,1] == 0)
    #     idx_bckg = np.where(idx_bckg.data.numpy())[0]
    #     length = int(float(idx_signal.shape[0])*mixing)
    #     idx_signal = idx_signal[:length]
    #     idx_bckg = idx_bckg[:length]
    #     mid = int(length/2.)
    #     for i in range(length):
    #         if i < mid:
    #             temp = X[idx_signal[i], 4:].clone()
    #             X[idx_signal[i], 4:] = X[idx_bckg[i], 4:]
    #             X[idx_bckg[i], 4:] = temp
    #         else:
    #             temp = X[idx_signal[i], :4].clone()
    #             X[idx_signal[i], :4] = X[idx_bckg[i], :4]
    #             X[idx_bckg[i], :4] = temp
    #     return X, y


    # mlp = copy_mlp(models['base-model'])
    # autoencoders = models['model-with'][(0.01,1,0)][0].autoencoders
    # min_potentials =  models['model-with'][(0.01,1,0)][0].min_potentials
    # emma = EMMA(n_modes, autoencoders, min_potentials).float()
    # model = nn.ModuleList([emma, mlp])
    # models['model-with'][(-1,-1,-1)] = model
    # meta['coldness'] = [0.01]
    # meta['lambda_regul'] = [1]
    # meta['lambda_capacity'] = [0]
    # meta['max_epochs'] = 20
    # for mixing in np.linspace(0,1,5):
    #     X, y = torch.zeros(X_train.size()).float(), torch.zeros(y_train.size()).float()
    #     X, y = mixup(X_train.clone(), y_train.clone(), indic_train, mixing)
    #     train_clf(models, 'model-with', meta, X, y, indic_train, X, y, indic_train)
    #     print(models['model-with'][(0.01,1,0)][0].get_gammas())
    #     models['model-with'][(-1,-1,-1)] = copy_emma(models['model-with'][(0.01,1,0)])


















