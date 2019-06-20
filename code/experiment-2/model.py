"""
    ...
"""


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


#---------------
class Composer(nn.Module):
    """
    ...
    """

    def __init__(self, n_modes):
        super().__init__()
        self.n_modes = n_modes
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.t = 1./np.sqrt(n_modes)
        self.mode_min = torch.empty(n_modes).float()
        self.Gamma = torch.nn.Parameter(torch.empty(n_modes, n_modes))
        self.w = torch.nn.Parameter(torch.zeros(n_modes).float())
        self.init_params()

    def init_params(self):
        self.mode_min.fill_(float("Inf"))
        nn.init.uniform_(self.Gamma, a=0., b=1.)

    def reset(self, n_steps):
        self.n_steps = n_steps
        self.step = 0
        self.t = 0
        return self

    def cooldown(self):
        rate = 1./np.sqrt(self.n_modes) / self.n_steps
        self.step += 1
        if self.step > self.n_steps:
            return self
        self.t = rate * self.step
        return self

    def correction(self, phi, train):
        if train:
            batch_min = torch.min(phi, dim=0)[0]
            self.mode_min = np.minimum(self.mode_min, batch_min)
        phi = phi - self.mode_min.unsqueeze(0) + np.exp(1)
        phi[phi < np.exp(1)] = np.exp(1)
        return phi

    def get_gamma(self, i, j):
        if i < j: 
            return self.Gamma[i,j]
        return self.Gamma[j,i]

    def compute_partial(self, phi):
        N, M = phi.size()
        PE = torch.zeros(N, M, M)
        for i in range(M):
            for j in range(M):
                g = self.get_gamma(i,j)
                if i == j:
                    PE[:,i,j] = (1+self.w[i])*phi[:,i]
                elif i < j:
                    PE[:,i,j] = torch.pow(phi[:,i], g) * torch.pow(phi[:,j], 1-g)
                else:
                    PE[:,i,j] = torch.pow(phi[:,i], g) * torch.pow(phi[:,j], 1-g)
        return PE

    def forward(self, phi, train=True):
        assert phi.size(-1) == self.n_modes
        phi = self.correction(phi, train)
        partial_energies = self.compute_partial(phi)
        mode_energies = torch.sum(partial_energies, dim=-1)
        logits = self.logsoftmax(-self.t * mode_energies)
        alphas = torch.exp(logits)
        return alphas, logits


















