import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


# ---------------
class DenoisingAutoEncoder(nn.Module):
    """
        - d_input: dimension of input 
        - d_hidden: number of hidden units 
        - noise: standard deviation of Gaussian white noise of corruption process
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

    """
        - x: input tensor, size N x D
                + N: batch size
                + D: dimension of each input sample
        - add_noise: boolean value, if true the model applies corruption to the 
                     input x
    """
    def forward(self, x, add_noise):
        assert x.size(1) == self.d_input
        if add_noise:
            x = self.corruption(x)  
        u = torch.sigmoid(self.encoder(x) + self.bias_h)  
        output = F.linear(u, self.encoder.weight.t()) + self.bias_r  
        return output


        

    







