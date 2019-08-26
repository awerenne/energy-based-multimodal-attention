"""
    Helper functions. Mainly plotting of measures.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
plt.style.use('seaborn-poster') #sets the size of the charts
plt.style.use('ggplot')
import sys
sys.path.append('../data/')
from data import noise_power


# ---------------
def plot_curves(curves, save=False, fname=None):
    """ Plot train- and validation curves """
    train_curve, valid_curve = curves
    train_curve = np.asarray(train_curve)
    valid_curve = np.asarray(valid_curve)
    epochs = np.arange(len(train_curve))
    plt.figure()
    plt.plot(epochs, train_curve, label="Train")
    plt.plot(epochs, valid_curve, label="Validation")
    plt.xlabel('Epochs', fontsize=18)
    plt.ylabel('MSE loss', fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=11)
    plt.legend()
    if save: plt.savefig(fname)
    plt.show()


# ---------------
def plot_signal_bckg(measures, save=False, fname=None):
    q_seen, q_unseen = measures
    plt.figure()
    labels = ['positive', 'negative']
    plt.boxplot([q_seen, q_unseen])
    plt.xticks(np.arange(1,3), labels, fontsize=25)
    plt.ylabel(r'$\Psi$', fontsize=30)
    plt.tick_params(axis='both', which='major', labelsize=25)
    if save: plt.savefig(fname)
    plt.show()
    

# ---------------
def plot_noisy_signal(measures, save=False, fname=None):
    snr = measures[:,0]
    potential_mean = measures[:,1]
    potential_std = measures[:,2]
    plt.figure()
    plt.fill_between(snr, potential_mean-2*potential_std, potential_mean+2*potential_std)
    plt.plot(snr, potential_mean, color='black')
    plt.xlabel(r'$\sigma$ corruption', fontsize=30)
    plt.ylabel(r'$\Psi$', fontsize=30)
    plt.tick_params(axis='both', which='major', labelsize=25)
    if save: plt.savefig(fname)
    plt.show()
    





























