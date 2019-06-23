"""
    Helper functions. Mainly plotting of measures.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
plt.style.use('seaborn-whitegrid')


# ---------------
def plot_noise_quantifier(measures, save=False):
    noise = measures[:,0,0]
    recon_mean = measures[:,1,0]
    recon_std = measures[:,2,0]
    plt.figure()
    plt.fill_between(noise, recon_mean-2*recon_std, recon_mean+2*recon_std)
    plt.plot(noise, recon_mean, c='black')
    plt.xlabel('Noise', fontsize=18)
    plt.ylabel('Reconstruction norm', fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=11)
    plt.show()
    if save: plt.savefig('results/noise_vs_reconst')

    noise = measures[:,0,1]
    potential_mean = measures[:,1,1]
    potential_std = measures[:,2,1]
    plt.figure()
    plt.fill_between(noise, potential_mean-2*potential_std, potential_mean+2*potential_std)
    plt.plot(noise, potential_mean, c='black')
    plt.xlabel('Noise', fontsize=18)
    plt.ylabel('Potential', fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=11)
    plt.show()
    if save: plt.savefig('results/noise_vs_potential')


# ---------------
def plot_seen_unseen(measures, save=False):
    q_seen, q_unseen = measures
    plt.figure()
    plt.boxplot([q_seen[:,0], q_seen[:,1], q_unseen[:,0], q_unseen[:,1]])
    labels = ['signal (potential)', 'signal (reconst.)', 'background (potential)',
                'background (reconst.)']
    plt.xticks(np.arange(1,4), labels)
    plt.ylabel('q', fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=11)
    plt.show()
    if save: plt.savefig('results/seen_vs_unseen')


# ---------------
def add_noise(X, noise):
    return X + Variable(X.data.new(X.size()).normal_(0, noise))

























