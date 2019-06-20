"""
    ...
"""


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------
class AutoEncoder(nn.Module):
    """
    ...
    """

    def __init__(self, d_input, d_hidden):
        super().__init__()
        self.d_input = d_input
        self.d_hidden = d_hidden
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.encoder = nn.Linear(d_input, d_hidden, bias=False)
        self.bias_h = torch.nn.Parameter(torch.zeros(d_hidden).float())
        self.bias_r = torch.nn.Parameter(torch.zeros(d_input).float())

    def derivative_sigmoid(self, x):
        x = self.sigmoid(x).squeeze()
        return x * (torch.ones(x.size()) - x)  # Element-wise product 

    def derivative_tanh(self, x):
        x = self.tanh(x).squeeze()
        return (torch.ones(x.size()) - x.pow(2))  # Element-wise product 

    def laplacian(self, x):
        W = self.encoder.weight
        deriv = self.derivative_sigmoid((self.encoder(x) + self.bias_h))
        laplacian = torch.trace(W.t() @ torch.diag(deriv) @ W) - self.d_input
        return -laplacian

        # laplacian = 0
        # x = x.squeeze()
        # u= self.encoder(x) + self.bias_h
        # for i in range(self.d_hidden):
        #     w = W[i,:].squeeze()
        #     laplacian += self.derivative_sigmoid(torch.dot(w,x) + self.bias_h[i])*w.norm(p=2).pow(2)
        #     # laplacian += self.derivative_tanh(torch.dot(w,x) + self.bias_h[i])*w.norm(p=2).pow(2)
        #     # laplacian += self.derivative_tanh(u.squeeze()[i])*w.norm(p=2).pow(2)
        # return laplacian - self.d_input

    def energy(self, x):
        energy = 0
        W = self.encoder.weight
        for i in range(self.d_hidden):
            e_unit = torch.exp(torch.matmul(x, W.t()[:,i]) + self.bias_h[i])
            energy += torch.log(1 + e_unit)
        energy -= 0.5 * (x - self.bias_r).norm(p=2).pow(2)
        return -energy

    def forward(self, x):
        u = self.sigmoid(self.encoder(x) + self.bias_h)
        # u = torch.tanh(self.encoder(x) + self.bias_h)
        output = F.linear(u, self.encoder.weight.t()) + self.bias_r
        return output


# m = AutoEncoder(2,5).laplacian(torch.ones(1,2))















