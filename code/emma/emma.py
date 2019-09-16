import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from autoencoder import DenoisingAutoEncoder


# ---------------
class EMMA(nn.Module):
    
    def __init__(self, n_modes, autoencoders, vmin, tau=None):
        super().__init__()
        self._n_modes = n_modes
        self._autoencoders = autoencoders   
        self.vmin = vmin
        self.tau = tau  
        
        self.w = torch.nn.Parameter(torch.ones(n_modes).float())
        self.W_coupling = torch.nn.Parameter(torch.ones(n_modes, n_modes).float())
        self.W_coupling.data.uniform_(-1.0/(n_modes-1), 1.0/(n_modes-1))
        self.bias_correction = torch.nn.Parameter(torch.zeros(n_modes).float())
        self.gammas = torch.nn.Parameter(torch.ones(n_modes, n_modes)) 
        self.gammas.data *= 0.5 
        self.gain = torch.nn.Parameter(torch.ones(1).float())
        self.bias_attention = torch.nn.Parameter(torch.zeros(1).float())

    # -------
    def set_coldness(self, tau):
        self.tau = tau

    # -------
    def get_coldness(self):
        return self.tau

    # -------
    def get_gain(self):
        return self.gain

    # -------
    def get_bias_attention(self):
        return self.bias_attention

    # -------
    def get_alpha_beta(self, x):
        N = x[0].size(0)
        potentials = torch.zeros(N, self._n_modes)
        for i in range(self._n_modes): 
            potentials[:,i] = self._autoencoders[i].potential(x[i])
        alphas, _, _ = self.compute_importance_scores(potentials)  # N x M
        _, betas = self.apply_attention(x, alphas)
        return alphas, betas

    # -------
    @property
    def capacity(self):
        return 1/self.gain * torch.log(torch.cosh(self.gain - self.bias_attention)/torch.cosh(-self.bias_attention))

    # -------
    def get_coupling(self):
        return (self.gammas, self.W_coupling)

    # -------
    def total_energy(self, x):
        N = x[0].size(0)
        potentials = torch.zeros(N, self._n_modes)
        for i in range(self._n_modes): 
            potentials[:,i] = self._autoencoders[i].potential(x[i])
        _, _, total_energy = self.compute_importance_scores(potentials)  # N x M
        return total_energy

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
    def correction(self, v):
        v = v - self.vmin.unsqueeze(0) + np.exp(1)
        v[v < np.exp(1)] = np.exp(1)
        return v * self.w + self.bias_correction

    # -------
    def gamma(self, i=None, j=None):
        if i == None or j == None: return torch.triu(self.gammas, diagonal=1)
        return self.gammas[i,j]

    # -------
    def compute_partial(self, v):
        partial_energies = torch.zeros(v.size(0), self._n_modes, self._n_modes)
        for i in range(self._n_modes):
            for j in range(self._n_modes):
                gamma_ij = self.gamma(i,j)
                if i == j:
                    partial_energies[:,i,j] = v[:,i]
                else:
                    partial_energies[:,i,j] = self.W_coupling[i,j] * \
                                torch.pow(v[:,i], gamma_ij) * \
                                torch.pow(v[:,j], 1-gamma_ij)
        return partial_energies

    # -------
    def compute_importance_scores(self, potentials):
        potentials = self.correction(potentials)  # N x M
        partial_energies = self.compute_partial(potentials)  # N x M x M
        modal_energies = torch.sum(partial_energies, dim=-1)  # N x M
        logs = F.log_softmax(-self.tau * modal_energies, dim=-1)  
        alphas = torch.exp(logs)
        return alphas, logs, torch.sum(modal_energies, dim=-1)

    # -------
    def apply_attention(self, x, alphas):
        betas = torch.relu(torch.tanh(self.gain * alphas - self.bias_attention))  
        xprime = []
        for i in range(self.n_modes):
            xprime.append(torch.mul(betas[:,i].unsqueeze(-1), x[i].clone()))
        return xprime, betas

    # -------
    def forward(self, x):
        assert len(x) == self._n_modes
        N = x[0].size(0)
        potentials = torch.zeros(N, self._n_modes)
        for i in range(self._n_modes): 
            potentials[:,i] = self._autoencoders[i].potential(x[i])
        alphas, logs, _ = self.compute_importance_scores(potentials)  # N x M
        xprime, _ = self.apply_attention(x, alphas)
        return xprime, logs


# ---------------
class WeightClipper(object):

    def __call__(self, emma):
        if hasattr(emma, 'w'): 
            emma.w.data = emma.w.data.clamp(min=1)

        if hasattr(emma, 'bias_correction'): 
            emma.bias_correction.data = emma.bias_correction.data.clamp(min=0)

        if hasattr(emma, 'gain'): 
            emma.gain.data = emma.gain.data.clamp(min=0)

        if hasattr(emma, 'bias_attention'): 
            emma.bias_attention.data = emma.bias_attention.data.clamp(min=0, max=1)

        if hasattr(emma, 'gammas'): 
            emma.gammas.data = emma.gammas.data.clamp(min=0, max=1)

        if hasattr(emma, 'W_coupling'): 
            emma.W_coupling.data = emma.W_coupling.data.clamp(min=-1, max=1)

    



    







