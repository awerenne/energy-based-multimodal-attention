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
        Small multi-layer perceptron used as the prediction model.
    """

    def __init__(self, d_input):
        super().__init__()
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
        Xhat = model(batch, add_noise=True)  
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
    dict_curves = {}
    dict_curves[name] = {}

    """ Train models without emma """
    if not name == 'model-with':
        n_epochs = meta['max_epochs']
        best_loss = float("Inf")
        best_model = None
        best_epoch = 0
        state_optim = None
        curves = np.zeros((2, n_epochs+1))
        model = models[name]
        optimizer = torch.optim.Adam(nn.ParameterList(model.parameters()))
        model.eval()
        with torch.set_grad_enabled(False):
            curves[1,0] = train_step(model, name, optimizer, meta['batch_size'], X_train, y_train, valid=True)     
        for epoch in range(1, n_epochs+1):
            model.train()
            curves[0,epoch] = train_step(model, name, optimizer, meta['batch_size'], X_train, y_train)
            model.eval()
            with torch.set_grad_enabled(False):
                curves[1,epoch] = train_step(model, name, optimizer, meta['batch_size'], X_train, y_train, valid=True)
            if curves[1,epoch] < best_loss:
                best_loss = curves[1,epoch]
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
        dict_curves[name] = curves
        return dict_curves
        
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
                curves = np.zeros((2, n_epochs+1))
                model = copy_emma(models[name][(-1,-1,-1)])
                model[0].set_coldness(tau)
                optimizer = torch.optim.Adam(nn.ParameterList(model.parameters()), lr=meta['lr'])
                model.eval()
                with torch.set_grad_enabled(False):
                    curves[1,0] = train_step(model, name, optimizer, meta['batch_size'], X_train, y_train, indic_train, lambda_regul, lambda_capacity, valid=True)
                for epoch in range(1, n_epochs+1):
                    model.train()
                    curves[0,epoch] = train_step(model, name, optimizer, meta['batch_size'], X_train, y_train, indic_train, lambda_regul, lambda_capacity)
                    model.eval()
                    with torch.set_grad_enabled(False):
                        curves[1,epoch] = train_step(model, name, optimizer, meta['batch_size'], X_train, y_train, indic_train, lambda_regul, lambda_capacity, valid=True)
                    if curves[1,epoch] < best_loss:
                        best_loss = curves[1,epoch]
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
                dict_curves[name][tau, lambda_regul, lambda_capacity] = curves
    del models[name][(-1,-1,-1)]
    return dict_curves


# ---------------
def train_step(model, name, optimizer, batch_size, X, y, indic=None, lambda_regul=None,
                lambda_capacity=None, valid=False):
    n_steps, sum_loss = 0, 0
    # criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.]))
    criterion = torch.nn.BCEWithLogitsLoss()
    indices = np.arange(X.size(0))
    np.random.shuffle(indices)
    for i in range(0, len(X)-batch_size, batch_size):
        optimizer.zero_grad()
        idx = indices[i:i+batch_size]
        batch = X[idx].view(batch_size, -1)
        if not name == 'model-with':
            yhat, logits = model(batch)
            loss = criterion(logits, y[idx].unsqueeze(-1))
        else:
            modes = [batch[:,:4], batch[:,4:]]
            modes, log_alphas = model[0](modes)
            new_batch = torch.zeros(batch.size()).float()
            new_batch[:, :4], new_batch[:, 4:]  = modes[0], modes[1]
            yhat, logits = model[1](new_batch)  
            loss = criterion(logits, y[idx].unsqueeze(-1))
            loss -= lambda_regul * regul_loss(indic[idx], log_alphas)
            loss += (lambda_capacity * (model[0].get_gain() - model[0].get_bias_attention()))[0]
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
    retrain = False
    n_modes = 2
    d_input = [4, 4]
    n_hidden = 12  # Number of hidden units in autoencoders
    noise_std_autoenc = 0.01
    noise_std_data = 0.5
    coldness = [1e-4, 1e-3, 1e-2, 1e-1, 1]
    lambda_regul = [0, 1e-4, 1e-3, 1e-2, 1e-1]
    lambda_capacity = [0, 1e-4, 1e-3, 1e-2, 1e-1]

    meta = {}
    meta['clipper'] = WeightClipper()
    meta['coldness'] = coldness
    meta['lambda_regul'] = lambda_regul
    meta['lambda_capacity'] = lambda_capacity
    meta['max_epochs'] = 35
    meta['batch_size'] = 32
    meta['lr'] = 1e-3

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
    autoencoders = {'IP': None, 'DM-SNR': None}
    min_potentials = [None, None]
    separate = []
    models = {'base-model': None, 'model-without': None, 'model-with': {(-1,-1,-1): None}}
    curves = {'base-model': None, 'model-without': None, 'model-with': {(-1,-1,-1): None}}
    if retrain:
        """ Train autoencoders on normal signal train-set """
        X_signal_train = X_train
        X_signal_valid = X_valid
        model = DenoisingAutoEncoder(d_input[0], n_hidden, noise_std_autoenc).float()
        optimizer = torch.optim.Adam(nn.ParameterList(model.parameters()))
        model, _ = train_autoencoder(model, optimizer, meta['batch_size'], 30, X_signal_train[:,:4], X_signal_valid[:,:4])
        autoencoders['IP'] = model
        min_potentials[0] = get_min_potential(X_signal_train[:,:4], model)
        torch.save((model.state_dict(), min_potentials[0]), "dumps/autoencoder-ip.pt")

        model = DenoisingAutoEncoder(d_input[1], n_hidden, noise_std_autoenc).float()
        optimizer = torch.optim.Adam(nn.ParameterList(model.parameters()))
        model, _ = train_autoencoder(model, optimizer, meta['batch_size'], 30, X_signal_train[:,4:], X_signal_valid[:,4:])
        autoencoders['DM-SNR'] = model
        min_potentials[1] = get_min_potential(X_signal_train[:,4:], model)
        min_potentials = torch.tensor(min_potentials).float()
        torch.save((model.state_dict(), min_potentials[1]), "dumps/autoencoder-dm-snr.pt")

        """ Freeze autoencoders """
        for key, autoencoder in autoencoders.items():
            freeze(autoencoder)

        """ Train base model on normal train-set, eval on valid-set """
        model = MLP(d_input=np.sum(d_input)).float()
        models['base-model'] = model
        curves = train_clf(models, 'base-model', meta, X_train, y_train, indic_train, 
            X_valid, y_valid, indic_valid)
        torch.save((curves), "dumps/curve-base.pt")

        """ Train model without EMMA noisy train-set, eval on noisy valid-set """
        mlp = copy_mlp(models['base-model'])
        models['model-without'] = mlp
        curves = train_clf(models, 'model-without', meta, X_train_noisy, y_train_noisy, indic_train, 
            X_valid_noisy, y_valid_noisy, indic_valid)
        torch.save((curves), "dumps/curve-without.pt")

        """ Train model with EMMA noisy train-set, eval on noisy valid-set """
        mlp = copy_mlp(models['base-model'])
        emma = EMMA(n_modes, list(autoencoders.values()), min_potentials).float()
        model = nn.ModuleList([emma, mlp])
        models['model-with'][(-1,-1,-1)] = model
        curves = train_clf(models, 'model-with', meta, X_train_noisy, y_train_noisy, indic_train, 
            X_valid_noisy, y_valid_noisy, indic_valid)
        torch.save((curves), "dumps/curve-with.pt")

        """ Save """
        torch.save((models), "dumps/models")
        torch.save((X_test, y_test, X_test_noisy, y_test_noisy, indic_test), "dumps/test-set.pt")

    else:
        """ Load autoencoders """
        autoencoders['IP'] = DenoisingAutoEncoder(d_input[0], n_hidden, noise_std_autoenc).float()
        params, min_potentials[0] = torch.load("dumps/autoencoder-ip.pt")
        autoencoders['IP'].load_state_dict(params)

        autoencoders['DM-SNR'] = DenoisingAutoEncoder(d_input[1], n_hidden, noise_std_autoenc).float()
        params, min_potentials[1] = torch.load("dumps/autoencoder-dm-snr.pt")
        autoencoders['DM-SNR'].load_state_dict(params)
        min_potentials = torch.tensor(min_potentials).float()

        """ Load models """
        models = torch.load("dumps/models")
        curves = torch.load("dumps/curve-with.pt")

        # """ Load noisy test-set (with indicator) """
        X_test, y_test, X_test_noisy, y_test_noisy, indic_test = torch.load("dumps/test-set.pt")
    
    def get_ranking(models, n_top, X, y, indic):
        if models is None: 
            return None
        ranking_f1, ranking_params = [], []
        for params, model in models.items():
            if model is None: continue
            model.eval()
            F1, _, _, _ = evaluation(model, 'model-with', X, y)
            ranking_f1.append(-F1)
            ranking_params.append(params)
        idx = np.argsort(ranking_f1)
        n_top = min(n_top, len(idx))
        ranking = []
        for i in range(n_top):
            ranking.append(ranking_params[idx[i]])
        return ranking

    # X_test_noisy, y_test_noisy, indic_test = apply_corruption(X_test, y_test,
    #                                                 0.5)  
    ranking = get_ranking(models['model-with'], 10, X_test_noisy, y_test_noisy, indic_test)
    print_evaluation(models, ranking, X_test_noisy, y_test_noisy, indic_test)

    """ Confusion matrix of best models """
    yhat, classes = predictions(models['base-model'], 'base-model', X_test_noisy) 
    print_confusion_matrix(classes.data.numpy(), y_test_noisy.data.numpy())
    refined_matrix(y_test_noisy, classes, indic_test)
    yhat, classes = predictions(models['model-without'], 'model-without', X_test_noisy) 
    print_confusion_matrix(classes.data.numpy(), y_test_noisy.data.numpy())
    refined_matrix(y_test_noisy, classes, indic_test)
    yhat, classes = predictions(models['model-with'][ranking[0]], 'model-with', X_test_noisy) 
    print_confusion_matrix(classes.data.numpy(), y_test_noisy.data.numpy())
    refined_matrix(y_test_noisy, classes, indic_test)

    X_test_noisy, y_test_noisy, indic_test = apply_corruption(X_test, y_test, 2)  
    # for i in range(10): 
    #     plot_distribution(models['model-with'][ranking[i]], X_test_noisy, indic_test, save=True, idx=i)
    plot_distribution(models['model-with'][ranking[0]], X_test_noisy, indic_test, save=True)
    # for i in range(10):
    #     plot_noise_generalisation(models, ranking[i], X_test, y_test, save=False, idx=i)
    plot_noise_generalisation(models, ranking[0], X_test, y_test, save=False, idx=0)
    plot_total_energy(models, ranking[0], X_test, y_test, save=True)



















