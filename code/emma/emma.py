import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from autoencoder import DenoisingAutoEncoder


# ---------------
class EMMA(nn.Module):
    def __init__(self, n_modes, autoencoders, min_pot_energies, rho=None):
        super().__init__()
        self._n_modes = n_modes
        self._autoencoders = autoencoders   
        self.min_pot_energies = min_pot_energies
        self.rho = rho  
        self.w_self = torch.nn.Parameter(torch.ones(n_modes).float())
        self.b_self = torch.nn.Parameter(torch.zeros(n_modes).float())
        self.W_coupling = torch.nn.Parameter(torch.ones(n_modes, n_modes).float())
        self.W_coupling.data.uniform_(-1.0/(n_modes-1), 1.0/(n_modes-1))
        self.gammas = torch.nn.Parameter(torch.ones(n_modes, n_modes)) 
        self.gammas.data *= 0.5 
        self.g_a = torch.nn.Parameter(torch.ones(1).float())
        self.b_a = torch.nn.Parameter(torch.zeros(1).float())

    # -------
    @property
    def capacity(self):
        return 1/self.g_a * torch.log(torch.cosh(self.g_a - self.b_a)/torch.cosh(-self.b_a))

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
        return self.min_pot_energies

    # -------
    def get_coldness(self):
        return self.rho

    # -------
    def get_g_a(self):
        return self.g_a

    # -------
    def get_b_a(self):
        return self.b_a

    # -------
    def get_alpha_beta(self, x):
        N = x[0].size(0)
        potentials = torch.zeros(N, self._n_modes)
        for i in range(self._n_modes): 
            potentials[:,i] = self._autoencoders[i].potential(x[i])
        alphas, _ = self.compute_importance_scores(potentials) 
        _, betas = self.apply_attention(x, alphas)
        return alphas, betas

    # -------
    def get_coupling(self):
        return (self.gammas, self.W_coupling)

    # -------
    def total_energy(self, x):
        N = x[0].size(0)
        potentials = torch.zeros(N, self._n_modes)
        for i in range(self._n_modes): 
            potentials[:,i] = self._autoencoders[i].potential(x[i])
        _, _, total_energy = self.compute_importance_scores(potentials) 
        return total_energy

    # -------
    def correction(self, potentials):
        potentials = potentials - self.min_pot_energies.unsqueeze(0) + np.exp(1)
        potentials[potentials < np.exp(1)] = np.exp(1)
        return potentials
        
    # -------
    def compute_self_energies(self, potentials):
        return potentials * self.w_self + self.b_self

    # -------
    def compute_shared_energies(self, self_energies):
        partial_energies = torch.zeros(self_energies.size(0), self._n_modes, self._n_modes)
        for i in range(self._n_modes):
            for j in range(self._n_modes):
                if i == j:
                    partial_energies[:,i,j] = self_energies[:,i]
                else:
                    partial_energies[:,i,j] = self.W_coupling[i,j] * \
                                torch.pow(self_energies[:,i], self.gammas[i,j]) * \
                                torch.pow(self_energies[:,j], 1-self.gammas[i,j])
        return partial_energies

    # -------
    def compute_importance_scores(self, potentials):
        potentials = self.correction(potentials) 
        self_energies = self.compute_self_energies(potentials) 
        shared_energies = self.compute_shared_energies(self_energies)  
        modal_energies = torch.sum(shared_energies, dim=-1)  
        logs = F.log_softmax(-self.rho * modal_energies, dim=-1)  
        alphas = torch.exp(logs)
        return alphas, logs

    # -------
    def apply_attention(self, x, alphas):
        betas = torch.relu(torch.tanh(self.g_a * alphas - self.b_a))  
        xprime = []
        for i in range(self.n_modes):
            xprime.append(torch.mul(betas[:,i].unsqueeze(-1), x[i].clone()))
        return xprime, betas

    # -------
    def forward(self, x):
        assert len(x) == self._n_modes
        potentials = torch.zeros(x[0].size(0), self._n_modes)
        for i in range(self._n_modes): 
            potentials[:,i] = self._autoencoders[i].potential(x[i])
        alphas, logs = self.compute_importance_scores(potentials)  
        xprime, _ = self.apply_attention(x, alphas)
        return xprime, logs


# ---------------
class WeightClipper(object):

    def __call__(self, emma):
        if hasattr(emma, 'w_self'): 
            emma.w_self.data = emma.w_self.data.clamp(min=1)

        if hasattr(emma, 'b_self'): 
            emma.b_self.data = emma.b_self.data.clamp(min=0)

        if hasattr(emma, 'g_a'): 
            emma.g_a.data = emma.g_a.data.clamp(min=0)

        if hasattr(emma, 'b_a'): 
            emma.b_a.data = emma.b_a.data.clamp(min=0, max=1)

        if hasattr(emma, 'gammas'): 
            emma.gammas.data = emma.gammas.data.clamp(min=0, max=1)

        if hasattr(emma, 'W_coupling'): 
            emma.W_coupling.data = emma.W_coupling.data.clamp(min=-1, max=1)

    



    







