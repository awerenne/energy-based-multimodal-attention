"""
    ...
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
import seaborn as sns
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
plt.style.use('seaborn-poster') #sets the size of the charts
plt.style.use('ggplot')
from sklearn.metrics import confusion_matrix
from main import evaluation, model_evaluation
from data import white_noise
import torch


# ---------------
def print_confusion_matrix(yhat, y):
    # True on row, predicted on columns
    cm = confusion_matrix(y, yhat)
    classes = [0, 1]
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm)
    print()


# ---------------
def plot_curves(curves, params, save=False):
    train_curve, valid_curve = curves['model-with'][params][0,:], curves['model-with'][params][1,:]
    epochs = np.arange(len(train_curve))
    plt.plot(epochs, train_curve, label="Train")
    plt.plot(epochs, valid_curve, label="Validation")
    plt.xlabel('epochs', fontsize=18)
    plt.ylabel('cross-entropy loss', fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=11)
    plt.legend()
    if save: plt.savefig('results/curves')
    plt.show()


# ---------------
def plot_eda():
    pass
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


# ---------------
def refined_matrix(y, classes, indic):
    mat = np.zeros((6,2))
    mat[0,0] = int(((y == 1) * (classes == 1) * (indic[:,0] == 0) * (indic[:,1] == 0)).sum().data.numpy())
    mat[0,1] = int(((y == 1) * (classes == 0) * (indic[:,0] == 0) * (indic[:,1] == 0)).sum().data.numpy())
    
    mat[1,0] = int(((y == 1) * (classes == 1) * (indic[:,0] == 1) * (indic[:,1] == 0)).sum().data.numpy())
    mat[1,1] = int(((y == 1) * (classes == 0) * (indic[:,0] == 1) * (indic[:,1] == 0)).sum().data.numpy())

    mat[2,0] = int(((y == 1) * (classes == 1) * (indic[:,0] == 0) * (indic[:,1] == 1)).sum().data.numpy())
    mat[2,1] = int(((y == 1) * (classes == 0) * (indic[:,0] == 0) * (indic[:,1] == 1)).sum().data.numpy())

    mat[3,0] = int(((y == 0) * (classes == 0) * (indic[:,0] == 0) * (indic[:,1] == 0)).sum().data.numpy())
    mat[3,1] = int(((y == 0) * (classes == 1) * (indic[:,0] == 0) * (indic[:,1] == 0)).sum().data.numpy())

    mat[4,0] = int(((y == 0) * (classes == 0) * (indic[:,0] == 1) * (indic[:,1] == 0)).sum().data.numpy())
    mat[4,1] = int(((y == 0) * (classes == 1) * (indic[:,0] == 1) * (indic[:,1] == 0)).sum().data.numpy())

    mat[5,0] = int(((y == 0) * (classes == 0) * (indic[:,0] == 0) * (indic[:,1] == 1)).sum().data.numpy())
    mat[5,1] = int(((y == 0) * (classes == 1) * (indic[:,0] == 0) * (indic[:,1] == 1)).sum().data.numpy())
    print(mat/np.sum(mat, axis=1, keepdims=True))
    print()


# ---------------
def plot_distribution(model, X, indic, save=False, idx=None):
    if idx is None:
        suffix = ""
    else: 
        suffix = "-" + str(idx)

    kde = False
    idx = (indic[:,0] == 0) * (indic[:,1] == 0)
    modes = [X[idx,:4], X[idx,4:]]
    alphas, _ = model[0].get_alpha_beta(modes)
    a = alphas.clone()
    # sns.distplot(alphas[:,0].data.numpy(), hist = True, kde = kde, rug=False, label = 'normal', color='blue')
    idx = (indic[:,0] == 1) * (indic[:,1] == 0)
    modes = [X[idx,:4], X[idx,4:]]
    alphas, _ = model[0].get_alpha_beta(modes)
    b = alphas.clone()
    # sns.distplot(alphas[:,0].data.numpy(), hist = True, kde = kde, rug=False, label = 'ip-noisy', color='red')
    idx = (indic[:,0] == 0) * (indic[:,1] == 1)
    modes = [X[idx,:4], X[idx,4:]]
    alphas, _ = model[0].get_alpha_beta(modes)
    c =alphas.clone()
    # sns.distplot(alphas[:,0].data.numpy(), hist = True, kde = kde, rug=False, label = 'dm-noisy', color='green')
    # plt.xlabel('alpha IP distribution', fontsize=17)
    # plt.tick_params(axis='both', which='major', labelsize=11)
    # plt.legend(loc='upper left')
    # if save: plt.savefig('results/alpha-distrib')
    # plt.show()

    barWidth = 0.25
    r1 = np.arange(3)
    r2 = [x + barWidth for x in r1]
     
    plt.figure() 
    plt.bar(r1, [torch.mean(a[:,0]).data.numpy(), torch.mean(b[:,0]).data.numpy(), torch.mean(c[:,0]).data.numpy()], width=barWidth, edgecolor='white', label=r'$\alpha$-ip')
    plt.bar(r2, [torch.mean(a[:,1]).data.numpy(), torch.mean(b[:,1]).data.numpy(), torch.mean(c[:,1]).data.numpy()], width=barWidth, edgecolor='white', label=r'$\alpha$-dm')
    plt.xticks([r + barWidth/2 for r in range(3)], ['uncorrupted', 'ip-noisy', 'dm-noisy'])
    for i, v in enumerate([torch.mean(a[:,0]).data.numpy(), torch.mean(b[:,0]).data.numpy(), torch.mean(c[:,0]).data.numpy()]):
        plt.text(i-0.07, v + .01, "%0.2f"%v, color='red', fontweight='bold', fontsize=21)
    for i, v in enumerate([torch.mean(a[:,1]).data.numpy(), torch.mean(b[:,1]).data.numpy(), torch.mean(c[:,1]).data.numpy()]):
        plt.text(i+0.17, v + .01, "%0.2f"%v, color='blue', fontweight='bold', fontsize=21)
    plt.legend(fontsize=19)
    plt.tick_params(axis='both', which='major', labelsize=25)
    plt.ylim([0, 1])
    if save: plt.savefig('results/noise-0-5/alpha-distrib' + suffix)
    plt.show()

    kde = False
    idx = (indic[:,0] == 0) * (indic[:,1] == 0)
    modes = [X[idx,:4], X[idx,4:]]
    _, betas = model[0].get_alpha_beta(modes)
    a = betas.clone()
    # sns.distplot(betas[:,0].data.numpy(), hist = True, kde = kde, rug=False, label = 'normal', color='blue')
    idx = (indic[:,0] == 1) * (indic[:,1] == 0)
    modes = [X[idx,:4], X[idx,4:]]
    _, betas = model[0].get_alpha_beta(modes)
    b = betas.clone()
    # sns.distplot(betas[:,0].data.numpy(), hist = True, kde = kde, rug=False, label = 'ip-noisy', color='red')
    idx = (indic[:,0] == 0) * (indic[:,1] == 1)
    modes = [X[idx,:4], X[idx,4:]]
    _, betas = model[0].get_alpha_beta(modes)
    c = betas.clone()
    # sns.distplot(betas[:,0].data.numpy(), hist = True, kde = kde, rug=False, label = 'dm-noisy', color='green')
    # plt.xlabel('beta IP distribution', fontsize=17)
    # plt.tick_params(axis='both', which='major', labelsize=11)
    # plt.legend(loc='upper left')
    # if save: plt.savefig('results/beta-distrib')
    # plt.show()

    plt.figure()
    plt.bar(r1, [torch.mean(a[:,0]).data.numpy(), torch.mean(b[:,0]).data.numpy(), torch.mean(c[:,0]).data.numpy()], width=barWidth, edgecolor='white', label=r'$\beta$-ip')
    plt.bar(r2, [torch.mean(a[:,1]).data.numpy(), torch.mean(b[:,1]).data.numpy(), torch.mean(c[:,1]).data.numpy()], width=barWidth, edgecolor='white', label=r'$\beta$-dm')
    plt.xticks([r + barWidth/2 for r in range(3)], ['uncorrupted', 'ip-noisy', 'dm-noisy'])
    for i, v in enumerate([torch.mean(a[:,0]).data.numpy(), torch.mean(b[:,0]).data.numpy(), torch.mean(c[:,0]).data.numpy()]):
        plt.text(i-0.07, v + .01, "%0.2f"%v, color='red', fontweight='bold', fontsize=21)
    for i, v in enumerate([torch.mean(a[:,1]).data.numpy(), torch.mean(b[:,1]).data.numpy(), torch.mean(c[:,1]).data.numpy()]):
        plt.text(i+0.17, v + .01, "%0.2f"%v, color='blue', fontweight='bold', fontsize=21)
    plt.legend(fontsize=19)
    plt.ylim([0, 1])
    if save: plt.savefig('results/noise-0-5/beta-distrib' + suffix)
    plt.show()


# ---------------
def plot_yerkes_dodson(coldness, models, best_combo, X, y, save=False):
    f1 = []
    for tau in coldness:
        model = models['model-with'][(tau, best_combo[1], best_combo[2])]
        F1, _, recall, _ = evaluation(model, 'model-with', X, y)
        f1.append(F1)
    plt.plot(coldness, f1, marker='o')
    plt.xscale('log')
    plt.xlabel('coldness', fontsize=17)
    plt.ylabel('F1-score', fontsize=17)
    plt.ylim([0, 1])
    plt.tick_params(axis='both', which='major', labelsize=11)
    plt.legend(loc='upper left')
    if save: plt.savefig('results/yerkes-dodson')
    plt.show()


# ---------------
def test(meta, models, X, y, save=False):
    f1 = []
    cap = []
    coldness = meta['coldness']
    lambda_regul = meta['lambda_regul'] 
    lambda_capacity = meta['lambda_capacity']
    for tau in coldness:
        best_F1 = -float("Inf")
        best_model = None
        for l_reg in lambda_regul:
            for l_cap in lambda_capacity:
                model = models['model-with'][(tau, l_reg, l_cap)]
                F1, _, _, _ = evaluation(model, 'model-with', X, y)
                if F1 > best_F1:
                    best_model = model
                    best_F1 = F1
        f1.append(best_F1)
        cap.append(best_model[0].capacity)
    plt.plot(coldness, f1, marker='o')
    plt.plot(coldness, cap, marker='o')
    plt.xscale('log')
    plt.xlabel('coldness', fontsize=17)
    plt.ylabel('F1-score/capacity', fontsize=17)
    plt.ylim([0, 1])
    plt.tick_params(axis='both', which='major', labelsize=11)
    plt.legend(loc='upper left')
    if save: plt.savefig('results/yo')
    plt.show()


# ---------------
def plot_capacity_vs_coldness(coldness, models, best_combo, X, y, save=False):
    cap = []
    for tau in coldness:
        model = models['model-with'][(tau, best_combo[1], best_combo[2])]
        cap.append(model[0].capacity)
    plt.plot(coldness, cap, marker='o')
    plt.xscale('log')
    plt.xlabel('coldness', fontsize=17)
    plt.ylabel('capacity', fontsize=17)
    plt.ylim([0, 1])
    plt.tick_params(axis='both', which='major', labelsize=11)
    plt.legend(loc='upper left')
    if save: plt.savefig('results/capacity-vs-coldness')
    plt.show()


# ---------------
def plot_noise_generalisation(models, best_combo, X_test, y_test, save=False, idx=None):
    if idx is None:
        suffix = ""
    else: 
        suffix = "-" + str(idx)
    # f1_with = []
    # f1_without = []
    # f1_base = []
    # noises = np.linspace(0,2,50)
    # for noise in noises:
    #     X = torch.zeros(X_test.size()).float()
    #     bigmid = int(X.size(0))
    #     smallmid = int(float(bigmid)/2)
    #     X = X_test.clone()
    #     X[:smallmid,:4] = white_noise(X[:smallmid,:4].data.numpy(), noise)
    #     X[smallmid:bigmid,4:] = white_noise(X[smallmid:bigmid,4:].data.numpy(), noise)
    #     F1, _, recall, _ = evaluation(models['model-with'][best_combo], 'model-with', X, y_test)
    #     f1_with.append(F1)
    #     F1, _, recall, _ = evaluation(models['model-without'], 'model-without', X, y_test)
    #     f1_without.append(F1)
    #     F1, _, recall, _ = evaluation(models['base-model'], 'base-model', X, y_test)
    #     f1_base.append(F1)
    # plt.plot(noises, f1_with, label='with')
    # plt.plot(noises, f1_without, label='without')
    # plt.plot(noises, f1_base, label='base')
    # plt.axvline(x=0.5, ls='--', c='green')
    # plt.xlabel(r'$\sigma$ corruption', fontsize=30)
    # plt.ylabel('F1-score', fontsize=30)
    # plt.ylim([0, 1])
    # plt.tick_params(axis='both', which='major', labelsize=25)
    # plt.legend(loc='best')
    # if save: plt.savefig('results/noise-generalisation-average' + suffix)
    # plt.show()

    # f1_with = []
    # f1_without = []
    # f1_base = []
    # noises = np.linspace(0,2,50)
    # beta_ip, beta_dm = [], []
    # for noise in noises:
    #     X = torch.zeros(X_test.size()).float()
    #     X = X_test.clone()
    #     X[:,:4] = white_noise(X[:,:4].data.numpy(), noise)
    #     F1, _, recall, _ = evaluation(models['model-with'][best_combo], 'model-with', X, y_test)
    #     f1_with.append(F1)
    #     F1, _, recall, _ = evaluation(models['model-without'], 'model-without', X, y_test)
    #     f1_without.append(F1)
    #     F1, _, recall, _ = evaluation(models['base-model'], 'base-model', X, y_test)
    #     f1_base.append(F1)
    #     modes = [X[:,:4], X[:,4:]]
    #     _, betas = models['model-with'][best_combo][0].get_alpha_beta(modes)
    #     beta_ip.append(torch.mean(betas[:,0]).data.numpy())
    #     beta_dm.append(torch.mean(betas[:,1]).data.numpy())
    # plt.plot(noises, f1_with, label='with')
    # plt.plot(noises, f1_without, label='without')
    # plt.plot(noises, f1_base, label='base')
    # plt.axvline(x=0.5, ls='--', c='green')
    # plt.xlabel(r'$\sigma$ corruption', fontsize=30)
    # plt.ylabel('F1-score', fontsize=30)
    # plt.ylim([0, 1])
    # plt.tick_params(axis='both', which='major', labelsize=25)
    # plt.legend(loc='best')
    # if save: plt.savefig('results/noise-generalisation-ip-noisy' + suffix)
    # plt.show()

    # plt.plot(noises, beta_ip, label=r'$\beta$-ip')
    # plt.plot(noises, beta_dm, label=r'$\beta$-dm')
    # plt.xlabel(r'$\sigma$ corruption', fontsize=30)
    # plt.ylabel(r'$\beta$', fontsize=30)
    # plt.ylim([-0.1, 1])
    # plt.tick_params(axis='both', which='major', labelsize=25)
    # plt.legend(loc='best')
    # if save: plt.savefig('results/noise-generalisation-ip-noisy-beta' + suffix)
    # plt.show()


    # beta_ip, beta_dm = [], []
    # f1_with = []
    # f1_without = []
    # f1_base = []
    # noises = np.linspace(0,2,50)
    # for noise in noises:
    #     X = torch.zeros(X_test.size()).float()
    #     X = X_test.clone()
    #     X[:,4:] = white_noise(X[:,4:].data.numpy(), noise)
    #     F1, _, recall, _ = evaluation(models['model-with'][best_combo], 'model-with', X, y_test)
    #     f1_with.append(F1)
    #     F1, _, recall, _ = evaluation(models['model-without'], 'model-without', X, y_test)
    #     f1_without.append(F1)
    #     F1, _, recall, _ = evaluation(models['base-model'], 'base-model', X, y_test)
    #     f1_base.append(F1)
    #     modes = [X[:,:4], X[:,4:]]
    #     _, betas = models['model-with'][best_combo][0].get_alpha_beta(modes)
    #     beta_ip.append(torch.mean(betas[:,0]).data.numpy())
    #     beta_dm.append(torch.mean(betas[:,1]).data.numpy())
    # plt.plot(noises, f1_with, label='with')
    # plt.plot(noises, f1_without, label='without')
    # plt.plot(noises, f1_base, label='base')
    # plt.axvline(x=0.5, ls='--', c='green')
    # plt.xlabel(r'$\sigma$ corruption', fontsize=30)
    # plt.ylabel('F1-score', fontsize=30)
    # plt.ylim([0, 1])
    # plt.tick_params(axis='both', which='major', labelsize=25)
    # plt.legend(loc='best')
    # if save: plt.savefig('results/noise-generalisation-dm-noisy' + suffix)
    # plt.show()

    # plt.plot(noises, beta_ip, label=r'$\beta$-ip')
    # plt.plot(noises, beta_dm, label=r'$\beta$-dm')
    # plt.xlabel(r'$\sigma$ corruption', fontsize=30)
    # plt.ylabel(r'$\beta$', fontsize=30)
    # plt.ylim([-0.1, 1])
    # plt.tick_params(axis='both', which='major', labelsize=25)
    # plt.legend(loc='best')
    # if save: plt.savefig('results/noise-generalisation-dm-noisy-beta' + suffix)
    # plt.show()


    # beta_ip, beta_dm = np.zeros((2,2)), np.zeros((2,2))
    # f1_with = []
    # f1_without = []
    # f1_base = []
    # noises = np.linspace(0,2,2)
    # for i, noise_ip in enumerate(noises):
    #     X = X_test.clone()
    #     X[:,:4] = white_noise(X[:,:4].data.numpy(), noise)
    #     for j, noise_dm in enumerate(noises):
    #         X[:,4:] = white_noise(X[:,4:].data.numpy(), noise)
    #         modes = [X[:,:4], X[:,4:]]
    #         _, betas = models['model-with'][best_combo][0].get_alpha_beta(modes)
    #         beta_ip[i,j] = torch.mean(betas[:,0]).data.numpy()
    #         beta_dm[i,j] = torch.mean(betas[:,1]).data.numpy()

    from mpl_toolkits.mplot3d import Axes3D
    x=np.linspace(0,2,20)
    y=np.linspace(0,2,20)
    lx= len(x)
    ly=len(y)
    beta_ip, beta_dm = np.zeros((20,20)), np.zeros((20,20))

    for i in range(lx):
        noise_ip = x[i]
        X = X_test.clone()
        X[:,:4] = white_noise(X[:,:4].data.numpy(), noise_ip)
        for j in range(ly):
            noise_dm = y[j]
            X[:,4:] = white_noise(X[:,4:].data.numpy(), noise_dm)
            modes = [X[:,:4], X[:,4:]]
            _, betas = models['model-with'][best_combo][0].get_alpha_beta(modes)
            beta_ip[i,j] = torch.mean(betas[:,0]).data.numpy()
            beta_dm[i,j] = torch.mean(betas[:,1]).data.numpy()

    fig = plt.figure()
    ax = Axes3D(fig)
    ax = fig.gca(projection='3d')
    x, y = np.meshgrid(x, y)
    surf = ax.plot_surface(x, y, beta_ip, rstride=1, cstride=1, cmap='hot')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

    fig = plt.figure()
    ax = Axes3D(fig)
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x, y, beta_dm, rstride=1, cstride=1, cmap='hot')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

    # from mpl_toolkits.mplot3d import Axes3D
    # import matplotlib.pyplot as plt
    # import numpy as np

    # X = np.arange(1, 2)
    # Y = np.arange(1, 2)
    # X, Y = np.meshgrid(X, Y)
    # R = np.sqrt(X**2 + Y**2)
    # Z = np.sin(R)
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='hot', linewidth=0, antialiased=False)
    # ax.set_zlim(-1.01, 1.01)

    # fig.colorbar(surf, shrink=0.5, aspect=5)
    # plt.show()


# ---------------
def plot_total_energy(models, best_combo, X_test, y_test, save=False):
    energy_system = []
    noises = np.linspace(0,2,50)
    f1 = []
    for noise in noises:
        X = torch.zeros(X_test.size()).float()
        bigmid = int(X.size(0)/2)
        smallmid = int(float(bigmid)/2)
        X = X_test.clone()
        X = white_noise(X.data.numpy(), noise)
        # X[:smallmid,:4] = white_noise(X[:smallmid,:4].data.numpy(), noise)
        # X[smallmid:bigmid,:4] = white_noise(X[smallmid:bigmid,:4].data.numpy(), noise)
        modes = [X[:,:4], X[:,4:]]
        energies = models['model-with'][best_combo][0].total_energy(modes).data.numpy()
        energy_system.append(np.mean(energies))
        F1, _, _, _ = evaluation(models['model-with'][best_combo], 'model-with', X, y_test)
        f1.append(F1)
    plt.plot(noises, energy_system)
    # plt.axvline(x=0.8, ls='--', c='red')
    plt.xlabel(r'$\sigma$ corruption', fontsize=30)
    plt.ylabel('Total Energy', fontsize=30)
    plt.tick_params(axis='both', which='major', labelsize=25)
    if save: plt.savefig('results/noise-vs-system-energy')
    plt.show()

    plt.plot(noises, f1)
    # plt.axvline(x=0.8, ls='--', c='red')
    plt.xlabel(r'$\sigma$ corruption', fontsize=30)
    plt.ylabel('F1-score', fontsize=30)
    plt.tick_params(axis='both', which='major', labelsize=25)
    if save: plt.savefig('results/noise-vs-system-f1')
    plt.show()


# ---------------
def plot_heatmap(models, meta, base_F1, cut_noise, X_test, y_test, save=False):
    X, y = torch.zeros(X_test.size()).float(), torch.zeros(y_test.size()).float()
    mid = int(X.size(0)/2)
    X[:mid, :4] = torch.tensor(white_noise(X_test[:mid, :4].data.numpy(), cut_noise)).float()
    X[mid:, 4:] = torch.tensor(white_noise(X_test[mid:, 4:].data.numpy(), cut_noise)).float()
    coldness = meta['coldness']
    lambda_regul = meta['lambda_regul']
    lambda_capacity = meta['lambda_capacity']
    score = np.zeros((len(lambda_regul),len(lambda_capacity)))
    for i, tau in enumerate(coldness):
        for j, l_cap in enumerate(lambda_capacity):
            best_F1 = -float("Inf")
            for l_reg in lambda_regul:
                F1, _, _, _ = evaluation(models['model-with'][(tau,l_reg,l_cap)], 'model-with', X, y_test)
                if F1 > best_F1:
                    best_F1 = F1
            score[i,j] = best_F1
    score = (score - base_F1)/base_F1
    ax = sns.heatmap(score, annot=True, xticklabels=lambda_capacity, yticklabels=coldness, cmap="YlGnBu")
    plt.xlabel('coldness', fontsize=17)
    plt.ylabel('lambda capacity', fontsize=17)
    if save: plt.savefig('results/heatmap')
    plt.show()


# ---------------
def print_evaluation(models, ranking, X, y, indic):
    print("F1-score -- Precision -- Recall -- Specificity")
    for name, model in models.items():
        print()
        print(name + ": ")
        if model is None: continue
        if name == 'model-with':
            for key in ranking:
                m = model[key]
                print(key)
                print("capacity: " + str(m[0].capacity.data.numpy()))
                m.eval()
                model_evaluation(m, name, X, y, indic)
            print()
        else:
            model.eval()
            model_evaluation(model, name, X, y, indic)
    print()

































