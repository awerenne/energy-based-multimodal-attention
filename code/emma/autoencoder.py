"""
    Implementation of a Denoising Autoencoder with potential energy computation
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


# ---------------
class DenoisingAutoEncoder(nn.Module):
    """
        Denoising Autoencoder.
        
        Inputs
            - x: input tensor, size N x D
                    + N: batch size
                    + D: dimension of each input sample
            - add_noise: boolean value, true forces the model to use the
                         corruption process

        Arguments
            - d_input: dimension of input (D)
            - d_hidden: number of hidden units (H)
            - noise: amplitude of noise in corruption process

    """

    def __init__(self, d_input, d_hidden, noise_stddev=0):
        super().__init__()
        self.d_input = d_input
        self.d_hidden = d_hidden
        self.noise_stddev = noise_stddev

        self.encoder = nn.Linear(d_input, d_hidden, bias=False)
        self.bias_h = torch.nn.Parameter(torch.zeros(d_hidden).float())
        self.bias_r = torch.nn.Parameter(torch.zeros(d_input).float())

    # -------
    def potential(self, x):
        potential = 0
        W = self.encoder.weight
        for i in range(self.d_hidden):
            e_unit = torch.exp(torch.matmul(x, W.t()[:,i]) + self.bias_h[i])
            potential -= torch.log(1 + e_unit)
        potential += 0.5 * (x - self.bias_r).norm(p=2).pow(2)
        return potential

    # -------
    def reconstruction_norm(self, x, add_noise=True):
        return (self.forward(x, add_noise)-x).norm(p=2).pow(2)

    # -------
    def corruption(self, x):
        if self.noise_stddev == 0: return x
        return x + Variable(x.data.new(x.size()).normal_(0, self.noise_stddev))

    # -------
    def forward(self, x, add_noise):
        assert x.size(1) == self.d_input
        if add_noise:
            x = self.corruption(x)  # N x D
        u = torch.sigmoid(self.encoder(x) + self.bias_h)  # N x H
        output = F.linear(u, self.encoder.weight.t()) + self.bias_r  # N x D
        return output


        

    







