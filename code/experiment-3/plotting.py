"""
    ...
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
import seaborn as sns
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
plt.style.use('seaborn-whitegrid')


# ---------------
def plot_1():
    """ Plot comparison increasing noise model_without model_with """
    pass


# ---------------
def plot_2(X, indic, model):
    """ Plot alpha beta on noisy test-set """
    idx = (indic[:,0] == 0) * (indic[:,1] == 0)
    modes = [X[idx,:4], X[idx,4:]]
    alphas, betas = model[0].get_alpha_beta(modes)
    sns.distplot(alphas[:,0].data.numpy(), hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3}, 
                  label = 'alphas')
    sns.distplot(betas[:,0].data.numpy(), hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3}, 
                  label = 'betas')
    # plt.legend(prop={'size': 16}, title = 'Airline')
    # plt.title('Density Plot with Multiple Airlines')
    # plt.xlabel('Delay (min)')
    # plt.ylabel('Density')
    plt.show()
    
    idx = (indic[:,0] == 1) * (indic[:,1] == 0)
    modes = [X[idx,:4], X[idx,4:]]
    alphas, betas = model[0].get_alpha_beta(modes)
    sns.distplot(alphas[:,0].data.numpy(), hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3}, 
                  label = 'alphas')
    sns.distplot(betas[:,0].data.numpy(), hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3}, 
                  label = 'betas')
    # plt.legend(prop={'size': 16}, title = 'Airline')
    # plt.title('Density Plot with Multiple Airlines')
    # plt.xlabel('Delay (min)')
    # plt.ylabel('Density')
    plt.show()
    
    idx = (indic[:,0] == 0) * (indic[:,1] == 1)
    modes = [X[idx,:4], X[idx,4:]]
    alphas, betas = model[0].get_alpha_beta(modes)
    sns.distplot(alphas[:,0].data.numpy(), hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3}, 
                  label = 'alphas')
    sns.distplot(betas[:,0].data.numpy(), hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3}, 
                  label = 'betas')
    # plt.legend(prop={'size': 16}, title = 'Airline')
    # plt.title('Density Plot with Multiple Airlines')
    # plt.xlabel('Delay (min)')
    # plt.ylabel('Density')
    plt.show()


# ---------------
def plot_3():
    """ Plot System Energy - F1-score on increasing noisy test-set on model_with and on model_without """
    pass


# ---------------
def plot_4(measures):
    """ Plot validation F1 (best temperature, threshold) of best lambda and lambda=0 """
    m_ = measures.data.numpy()[:,:,-1,1]  # last epoch
    idx = np.unravel_index(np.argmax(m_, axis=None), m_.shape)
    idx_best_temperature = idx[0]
    for i in range(measures.size(1)):
        f1_curve = measures[idx_best_temperature, i, -1, :].data.numpy()
        plt.plot(f1_curve)
    plt.show()


# ---------------
def plot_5(measures):
    """ Plot validation F1 (best lambda, threshold) of all temperatures """
    m_ = measures.data.numpy()[:,:,-1,1]  # last epoch
    idx = np.unravel_index(np.argmax(m_, axis=None), m_.shape)
    idx_best_lambda = idx[1]
    for i in range(measures.size(0)):
        f1_curve = measures[i, idx_best_lambda, 1, :].data.numpy()
        plt.plot(f1_curve)
    plt.show()


# ---------------
def plot_6(measures):
    """ Plot Yerkes-Dodson """
    m_ = measures.data.numpy()[:,:,-1,1]  # last epoch
    idx = np.unravel_index(np.argmax(m_, axis=None), m_.shape)
    idx_best_lambda = idx[1]
    f1_curve = measures[:, idx_best_lambda, 1, -1].data.numpy()
    coldness = np.asarray([0, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e2, 1e3, 1e4])
    plt.plot(np.log(coldness), f1_curve)
    plt.show()


# ---------------
def plot_7(measures):
    """ Plot capacity versus temperature """
    m_ = measures.data.numpy()[:,:,-1,1]  # last epoch
    idx = np.unravel_index(np.argmax(m_, axis=None), m_.shape)
    idx_best_lambda = idx[1]
    capacity = measures[:, idx_best_lambda, 3, -1].data.numpy()
    coldness = np.asarray([0, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e2, 1e3, 1e4])
    plt.plot(np.log(coldness), capacity)
    plt.show()






























