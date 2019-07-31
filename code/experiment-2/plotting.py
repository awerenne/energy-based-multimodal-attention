"""
    Helper functions. Mainly plotting of measures.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
plt.style.use('seaborn-whitegrid')
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
    labels = ['signal', 'background']
    plt.boxplot([q_seen, q_unseen])
    plt.xticks(np.arange(1,3), labels)
    plt.ylabel('Potential', fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=11)
    if save: plt.savefig(fname)
    plt.show()
    


# ---------------
def plot_noisy_signal(measures, save=False, fname=None):
    # noise_db = noise_power(measures[:,0])
    snr = 10 * np.log(1/(measures[:,0] ** 2))
    potential_mean = measures[:,1]
    potential_std = measures[:,2]
    plt.figure()
    plt.fill_between(snr, potential_mean-2*potential_std, potential_mean+2*potential_std)
    plt.plot(snr, potential_mean, color='black')
    plt.xlabel('SNR (dB)', fontsize=18)
    plt.ylabel('Potential', fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=11)
    if save: plt.savefig(fname)
    plt.show()
    





























