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
from sklearn.metrics import confusion_matrix


# ---------------
def plot_1():
    """ Plot comparison increasing noise model_without model_with """
    pass


# ---------------
def plot_confusion_matrix(yhat, y):
    cm = confusion_matrix(y, yhat)
    classes = [0, 1]
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm)

    # fig, ax = plt.subplots()
    # im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    # ax.figure.colorbar(im, ax=ax)
    # # We want to show all ticks...
    # ax.set(xticks=np.arange(cm.shape[1]),
    #        yticks=np.arange(cm.shape[0]),
    #        # ... and label them with the respective list entries
    #        xticklabels=classes, yticklabels=classes,
    #        title=title,
    #        ylabel='True label',
    #        xlabel='Predicted label')

    # # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    #          rotation_mode="anchor")

    # # Loop over data dimensions and create text annotations.
    # fmt = '.2f' if normalize else 'd'
    # thresh = cm.max() / 2.
    # for i in range(cm.shape[0]):
    #     for j in range(cm.shape[1]):
    #         ax.text(j, i, format(cm[i, j], fmt),
    #                 ha="center", va="center",
    #                 color="white" if cm[i, j] > thresh else "black")
    # fig.tight_layout()
    # return ax


# ---------------
def plot_2(model, X, indic):
    """ Plot alpha beta on noisy test-set """
    idx = (indic[:,0] == 0) * (indic[:,1] == 0)
    modes = [X[idx,:4], X[idx,4:]]
    alphas, _ = model[0].get_alpha_beta(modes)
    # sns.distplot(alphas[:,0].data.numpy(), hist = True, kde = False, rug=True, label = 'normal')
    idx = (indic[:,0] == 1) * (indic[:,1] == 0)
    modes = [X[idx,:4], X[idx,4:]]
    alphas, _ = model[0].get_alpha_beta(modes)
    sns.distplot(alphas[:,0].data.numpy(), hist = True, kde = False, rug=True, label = 'ip-noisy')
    idx = (indic[:,0] == 0) * (indic[:,1] == 1)
    modes = [X[idx,:4], X[idx,4:]]
    alphas, _ = model[0].get_alpha_beta(modes)
    sns.distplot(alphas[:,0].data.numpy(), hist = True, kde = False, rug=True, label = 'dm-noisy')
    plt.legend(loc='upper left')
    plt.show()

    idx = (indic[:,0] == 0) * (indic[:,1] == 0)
    modes = [X[idx,:4], X[idx,4:]]
    _, betas = model[0].get_alpha_beta(modes)
    # sns.distplot(betas[:,0].data.numpy(), hist = True, kde = False, rug=True, label = 'normal')
    idx = (indic[:,0] == 1) * (indic[:,1] == 0)
    modes = [X[idx,:4], X[idx,4:]]
    _, betas = model[0].get_alpha_beta(modes)
    sns.distplot(betas[:,0].data.numpy(), hist = True, kde = False, rug=True, label = 'ip-noisy')
    idx = (indic[:,0] == 0) * (indic[:,1] == 1)
    modes = [X[idx,:4], X[idx,4:]]
    _, betas = model[0].get_alpha_beta(modes)
    sns.distplot(betas[:,1].data.numpy(), hist = True, kde = False, rug=True, label = 'dm-noisy')
    plt.legend(loc='upper left')
    plt.show()


# ---------------
def plot_3():
    """ Plot System Energy - F1-score on increasing noisy test-set on model_with and on model_without """
    pass


# ---------------
def plot_4(measures):
    """ Plot validation F1 (best temperature, threshold) of best lambda and lambda=0 """
    m_ = measures.data.numpy()[:,:,-1,4,0]  # last epoch
    idx = np.unravel_index(np.argmax(m_, axis=None), m_.shape)
    idx_best_temperature = idx[0]
    for i in range(measures.size(1)):
        f1_curve = measures[idx_best_temperature, i, :, 4, 0].data.numpy()
        plt.plot(f1_curve)
    plt.show()


# ---------------
def plot_5(measures):
    """ Plot validation F1 (best lambda, threshold) of all temperatures """
    m_ = measures.data.numpy()[:,:,-1,4,1]  # last epoch
    idx = np.unravel_index(np.argmax(m_, axis=None), m_.shape)
    idx_best_lambda = idx[1]
    for i in range(measures.size(0)):
        f1_curve = measures[i, idx_best_lambda, :, 4, 0].data.numpy()
        plt.plot(f1_curve)
    plt.show()


# ---------------
def plot_6(measures):
    """ Plot Yerkes-Dodson """
    m_ = measures.data.numpy()[:,:,-1,4,1]  # last epoch
    idx = np.unravel_index(np.argmax(m_, axis=None), m_.shape)
    idx_best_lambda = idx[1]
    f1_curve = measures[:, idx_best_lambda, -1, 4, 0].data.numpy()
    coldness = np.asarray([0, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e2, 1e3, 1e4])
    plt.plot(np.log(coldness), f1_curve)
    plt.show()


# ---------------
def plot_7(measures):
    """ Plot capacity versus temperature """
    m_ = measures.data.numpy()[:,:,-1,4,1]  # last epoch
    idx = np.unravel_index(np.argmax(m_, axis=None), m_.shape)
    idx_best_lambda = idx[1]
    capacity = measures[:, idx_best_lambda, -1, 4, 4].data.numpy()
    coldness = np.asarray([0, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e2, 1e3, 1e4])
    plt.plot(np.log(coldness), capacity)
    plt.show()






























