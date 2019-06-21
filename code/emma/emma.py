"""
    ...
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ---------------
class EMMA(nn.Module):
    """
        ...
    """

    def __init__(self, n_modes, quantifiers, scheduler, energy=True):
        super().__init__()
        self.n_modes = n_modes
        self.quantifier = quantifier
        self.scheduler = scheduler

        self.qmin = torch.empty(n_modes).float()
        self.qmin.fill_(float("Inf"))
        
        self.w = torch.nn.Parameter(torch.zeros(n_modes).float())
        self.b = torch.nn.Parameter(torch.zeros(n_modes).float())
        self.gammas = torch.nn.Parameter(torch.ones(n_modes, n_modes))
        self.gammas.data *= 0.5 
        self.tau = 0  

        self.G = torch.nn.Parameter(torch.zeros(d_hidden).float())
        self.b = torch.nn.Parameter(torch.zeros(d_input).float())

    # -------
    def correction(self, q, train):
        if train:
            batch_min = torch.min(q, dim=0)[0]
            self.qmin = np.minimum(self.qmin, batch_min)
        q = q - self.qmin.unsqueeze(0) + np.exp(1)
        q[q < np.exp(1)] = np.exp(1)
        return q

    # -------
    def gamma(self, i=None, j=None):
        if i == None or j == None:
            return torch.triu(self.gammas, diagonal=1)
        if i < j: 
            return self.gammas[i,j]
        return self.gammas[j,i]

    # -------
    def to_partial(self, phi):
        N, M = phi.size()
        partials = torch.zeros(N, M, M)
        for i in range(M):
            for j in range(M):
                g = self.gamma(i,j)
                partials[:,i,j] = torch.pow(phi[:,i], g) * torch.pow(phi[:,j], 1-g)
        return partials

    # -------
    def to_modal(self, partial_energies):
        return torch.sum(partial_energies, dim=-1)

    # -------
    def fusion(self, q, train=True):
        phi = (1 + self.w) * self.correction(q, train).t() + self.b
        partial_energies = self.to_partial(phi)
        modal_energies = self.to_modal(partial_energies)
        logs = F.log_softmax(-self.tau * modal_energies, dim=-1)
        alphas = torch.exp(logs)
        return alphas, logs

    # -------
    def attention(self, x, alphas):
        betas = self.relu(self.tanh(self.G * alphas + self.bias))
        return betas * m

    # -------
    def forward(self, x):
        assert x.size(1) == self.n_modes
        q = torch.zeros(x.size(0), n_modes)
        for i in range(self.n_modes): 
            if energy: q[:,i] = self.quantifier.energy(x[:,i])
            else: q[:,i] = self.quantifier.reconstruction(x[:,i])
        alphas, _ = self.fusion(q)
        x = self.attention(x, alphas)
        return x
        




    



    







