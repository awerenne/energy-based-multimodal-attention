"""
    Implementation of the EMMA module.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from quantifier import *
from cooling import *


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
            - n_modes: number of modes of input
            - quantifiers: set of trained quantifiers
            - scheduler: cooling scheduler
    """

    def __init__(self, n_modes, quantifiers, qmin, scheduler):
        super().__init__()
        self.M = n_modes
        self.quantifiers = quantifiers   
        self.scheduler = scheduler
        self.qmin = qmin

        self.qmin = torch.empty(n_modes).float()
        self.qmin.fill_(float("Inf"))
        
        self.w = torch.nn.Parameter(torch.zeros(n_modes).float())
        self.b_f = torch.nn.Parameter(torch.zeros(n_modes).float())
        self.gammas = torch.nn.Parameter(torch.ones(n_modes, n_modes))
        self.gammas.data *= 0.5 
        self.tau = 1e-9  

        self.gain = torch.nn.Parameter(torch.ones(1).float())
        self.b_a = torch.nn.Parameter(torch.zeros(1).float())

    # -------
    def correction(self, q, train):
        # if train:
        #     batch_min = torch.min(q, dim=0)[0]
        #     self.qmin = np.minimum(self.qmin, batch_min)
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
    def fusion(self, q, train):
        phi = self.correction(q, train) * (1 + self.w) + self.b_f  # N x M
        partial_energies = self.to_partial(phi)  # N x M x M
        modal_energies = self.to_modal(partial_energies)  # N x M
        logs = F.log_softmax(-self.tau * modal_energies, dim=-1)  
        alphas = torch.exp(logs)
        return alphas, logs

    # -------
    def attention(self, x, alphas):
        betas = torch.relu(torch.tanh(self.gain * alphas + self.b_a))  # N x M
        xprime = []
        for i in range(self.M):
            xprime.append(torch.mul(betas[:,i].unsqueeze(-1), x[i].clone()))
        return xprime

    # -------
    def cooling(self):
        self.scheduler.cooldown()

    # -------
    def forward(self, x, train=True):
        assert len(x) == self.M
        N = x[0].size(0)
        q = torch.zeros(N, self.M)
        for i in range(self.M): 
            q[:,i] = self.quantifiers[i].energy(x[i])
        alphas, _ = self.fusion(q, train)  # N x M
        return self.attention(x, alphas)  # list of M tensors (N x D_i)
   

if __name__ == "__main__":
    x = []
    m = []
    N = 10
    M = 5
    for i in range(M):
        x.append(torch.ones(N, i+10))
        m.append(torch.ones(N, i+2))



    



    







