"""
    ...
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ---------------
class AutoEncoder(nn.Module):
    """
    ...
    """

    def __init__(self, d_input, d_hidden):
        super().__init__()
        self.d_input = d_input
        self.d_hidden = d_hidden
        self.encoder = nn.Linear(d_input, d_hidden, bias=False)
        self.bias_h = torch.nn.Parameter(torch.zeros(d_hidden).float())
        self.bias_r = torch.nn.Parameter(torch.zeros(d_input).float())

    def derivative_sigmoid(self, x):
        x = torch.sigmoid(x).squeeze()
        return x * (torch.ones(x.size()) - x)  # Element-wise product 

    def laplacian(self, x):
        x = x.squeeze()
        assert len(x.size()) == 1
        W = self.encoder.weight
        deriv = self.derivative_sigmoid((self.encoder(x) + self.bias_h))
        laplacian = torch.trace(W.t() @ torch.diag(deriv) @ W) - self.d_input
        return laplacian

    def energy(self, x):
        x = x.squeeze()
        assert len(x.size()) == 1
        energy = 0
        W = self.encoder.weight
        for i in range(self.d_hidden):
            e_unit = torch.exp(torch.matmul(x, W.t()[:,i]))
            energy += torch.log(1 + e_unit + self.bias_h[i])
        energy -= 0.5 * (x - self.bias_r).norm(p=2).pow(2)
        return energy

    def forward(self, x):
        u = torch.sigmoid(self.encoder(x) + self.bias_h)
        output = F.linear(u, self.encoder.weight.t()) + self.bias_r
        return output


# ---------------
class Composer(nn.Module):
    """
    ...
    """

    def __init__(self, n_modes):
        super().__init__()
        self.n_modes = n_modes
        self.tau = 1./np.sqrt(n_modes)
        self.Vmin = torch.empty(n_modes).float()
        self.w = torch.nn.Parameter(torch.zeros(n_modes).float())
        self.b = torch.nn.Parameter(torch.zeros(n_modes).float())
        self.gammas = torch.nn.Parameter(torch.ones(n_modes, n_modes))
        self.init_params()

    def init_params(self):
        self.Vmin.fill_(float("Inf"))
        self.gammas.data *= 0.5

    def reset(self, n_cooldown_steps):
        self.n_steps = n_cooldown_steps
        self.step = 0
        self.tau = 0
        return self

    def cooldown(self):
        # Linear cooldown
        max_tau = 1./np.sqrt(self.n_modes)
        step_size = max_tau / self.n_steps
        self.step += 1
        if self.step > self.n_steps:
            return self
        self.tau = step_size * self.step
        return self

    def correction(self, V, train):
        if train:
            batch_min = torch.min(V, dim=0)[0]
            self.Vmin = np.minimum(self.Vmin, batch_min)
        V = V - self.Vmin.unsqueeze(0) + np.exp(1)
        V[V < np.exp(1)] = np.exp(1)
        return V

    def get_gamma(self, i=None, j=None):
        if i == None or j == None:
            return torch.triu(self.gammas, diagonal=1)
        if i < j: 
            return self.gammas[i,j]
        return self.gammas[j,i]

    def compute_partial(self, phi):
        N, M = phi.size()
        partials = torch.zeros(N, M, M)
        for i in range(M):
            for j in range(M):
                gamma = self.get_gamma(i,j)
                if i == j:
                    partials[:,i,j] = phi[:,i]
                else:
                    partials[:,i,j] = torch.pow(phi[:,i], gamma) * \
                                        torch.pow(phi[:,j], 1-g)
        return partials

    def compute_modal(self, partial_energies):
        return torch.sum(partial_energies, dim=-1)

    def forward(self, V, train=True):
        assert V.size(-1) == self.n_modes
        print((1+self.w).size())
        print(self.correction(V, train).t().size())
        print(self.b.size())
        print(((1+self.w) @ self.correction(V, train).t()).size())
        phi = (1+self.w) * self.correction(V, train).t() + self.b
        partial_energies = self.compute_partial(phi)
        modal_energies = self.compute_modal(partial_energies)
        logits = F.log_softmax(-self.tau * modal_energies, dim=-1)
        alphas = torch.exp(logits)
        return alphas, logits


# ---------------
class EMMA(nn.Module):
    """
    ...
    """

    def __init__(self, n_modes):
        super().__init__()
        self.n_modes = n_modes
        self.autoencoder = AutoEncoder()
        self.composer = Composer()
        self.clf
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.G = torch.nn.Parameter(torch.zeros(d_hidden).float())
        self.b = torch.nn.Parameter(torch.zeros(d_input).float())

    def forward(self, m):
        V = self.autoencoder.energy(m)
        alphas = self.composer(V)
        betas = self.relu(self.tanh(self.G * alphas + self.bias))
        return betas * m


if __name__ == "__main__":
    # N = 10
    # d_input = 2
    # n_hidden = 5
    # x = torch.ones(N, d_input).float()
    # model = AutoEncoder(2,5).float()
    # model(x)
    # model.energy(x[0])
    # model.laplacian(x[0])

    N = 10
    n_modes = 3
    V = torch.ones(N, n_modes).float()
    model = Composer(n_modes).float()
    model(V)




    







