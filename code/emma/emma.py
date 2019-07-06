"""
    Implementation of the EMMA module.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from autoencoder import DenoisingAutoEncoder


# ---------------
class EMMA(nn.Module):
    """
        EMMA module. 

        Inputs
            - x: list of M input tensors, each tensor is of size N x ...
                    + M: number of modes
                    + N: batch size
            - m: list of M feature tensors, each tensor is of size N x D_i
                    + D_i: number of features in the i-th mode
            - train: boolean value, is true if training

        Arguments
            - ...
    """

    def __init__(self, n_modes, autoencoders, vmin, tau=None):
        super().__init__()
        self._n_modes = n_modes
        self._autoencoders = autoencoders   
        self.vmin = vmin
        self.tau = tau  
        
        self.w = torch.nn.Parameter(torch.ones(n_modes).float())
        self.bias_correction = torch.nn.Parameter(torch.zeros(n_modes).float())
        self.gammas = torch.nn.Parameter(torch.ones(n_modes, n_modes)) 
        self.gammas.data *= 0.5 
        self.gain = torch.nn.Parameter(torch.ones(1).float())
        self.bias_attention = torch.nn.Parameter(torch.zeros(1).float())

    # -------
    def set_coldness(self, tau):
        self.tau = tau

    # -------
    @property
    def n_modes(self):
        return self._n_modes

    # -------
    @property
    def autoencoders(self):
        return self._autoencoders

    # -------
    @property
    def min_potentials(self):
        return self.vmin

    # -------
    def correction(self, v, train):
        v = v - self.vmin.unsqueeze(0) + np.exp(1)
        v[v < np.exp(1)] = np.exp(1)
        return v * self.w + self.bias_correction

    # -------
    def gamma(self, i=None, j=None):
        if i == None or j == None: return torch.triu(self.gammas, diagonal=1)
        if i < j: return self.gammas[i,j]
        return self.gammas[j,i]

    # -------
    def compute_partial(self, v):
        partial_energies = torch.zeros(v.size(0), self._n_modes, self._n_modes)
        for i in range(self._n_modes):
            for j in range(self._n_modes):
                gamma_ij = self.gamma(i,j)
                partial_energies[:,i,j] = torch.pow(v[:,i], gamma_ij) * \
                                          torch.pow(v[:,j], 1-gamma_ij)
        return partial_energies

    # -------
    def compute_importance_scores(self, potentials, train):
        potentials = self.correction(potentials, train)  # N x M
        partial_energies = self.compute_partial(potentials)  # N x M x M
        modal_energies = torch.sum(partial_energies, dim=-1)  # N x M
        logs = F.log_softmax(-self.tau * modal_energies, dim=-1)  
        alphas = torch.exp(logs)
        return alphas, logs

    # -------
    def apply_attention(self, x, alphas):
        betas = torch.relu(torch.tanh(self.gain * alphas - self.bias_attention))  
        xprime = []
        for i in range(self.n_modes):
            xprime.append(torch.mul(betas[:,i].unsqueeze(-1), x[i].clone()))
        return xprime

    # -------
    def forward(self, x, train=True):
        assert len(x) == self._n_modes
        N = x[0].size(0)
        potentials = torch.zeros(N, self._n_modes)
        for i in range(self._n_modes): 
            potentials[:,i] = self._autoencoders[i].potential(x[i])
        alphas, logs = self.compute_importance_scores(potentials, train)  # N x M
        return self.apply_attention(x, alphas), logs


# ---------------
class WeightClipper(object):

    def __call__(self, emma):
        if hasattr(emma, 'w'): 
            param = emma.w.data
            param = param.clamp(min=1)

        if hasattr(emma, 'bias_correction'): 
            param = emma.bias_correction.data
            param = param.clamp(min=0)

        if hasattr(emma, 'gain'): 
            param = emma.gain.data
            param = param.clamp(min=0)

        if hasattr(emma, 'bias_attention'): 
            param = emma.bias_attention.data
            param = param.clamp(min=0, max=1)

        if hasattr(emma, 'gammas'): 
            param = emma.gammas.data
            param = param.clamp(min=0, max=1)

    



    







