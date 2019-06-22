"""
    ...
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
plt.style.use('seaborn-whitegrid')


# ---------------
def plot_noise_energy(measures):
    plt.figure()
    plt.fill_between(measures[:,0], measures[:,1]-2*measures[:,2],
            measures[:,1]+2*measures[:,2])
    plt.plot(measures[:,0], measures[:,1],c='black')
    plt.xlabel('Noise', fontsize=18)
    plt.ylabel('Reconstruction norm', fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=11)
    plt.show()
    plt.savefig('results/noise_vs_energy')


# ---------------
def plot_seen_unseen(measures):
    plt.figure()
    plt.boxplot([energies_0, energies_1])
    plt.xticks([1, 2], [0, 1])
    plt.xlabel('Mode', fontsize=18)
    plt.ylabel('Reconstruction norm', fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=11)
    plt.show()
    plt.savefig('results/seen_vs_unseen')


# ---------------
def add_noise(X, noise):
    pass
























