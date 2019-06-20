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

    def __init__(self, d_input, d_hidden, activation='sigmoid'):
        super().__init__()
        self.d_input = d_input
        self.d_hidden = d_hidden
        self.activation = activation
        self.encoder = nn.Linear(d_input, d_hidden, bias=False)
        self.bias_h = torch.nn.Parameter(torch.zeros(d_hidden).float())
        self.bias_r = torch.nn.Parameter(torch.zeros(d_input).float())

    def energy(self, x):
        energy = 0
        W = self.encoder.weight
        if self.activation == 'sigmoid':
            for i in range(self.d_hidden):
                e_unit = torch.exp(torch.matmul(x, W.t()[:,i]) + self.bias_h[i])
                energy -= torch.log(1 + e_unit)
            energy += 0.5 * (x - self.bias_r).norm(p=2).pow(2)
        elif self.activation == 'tanh':
            pass
        elif self.activation == 'relu':
            pass
        return energy

    def reconstruction(self, x):
        return (self.forward(x)-x).norm(p=2)

    def forward(self, x):
        x = self.encoder(x) + self.bias_h
        u = 0
        if self.activation == 'sigmoid':
            u = F.sigmoid(x)
        elif self.activation == 'tanh':
            u = F.tanh(x)
        elif self.activation == 'relu':
            u = F.relu(x)
        output = F.linear(u, self.encoder.weight.t()) + self.bias_r
        return output


# ---------------
if __name__ == "__main__":
    pass















