"""
    Implementation of two different quantifiers models: the denoising 
    autoencoder and the convolutional autoencoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from abc import ABC, abstractmethod
from cooling import Scheduler


# --------------- 
class Quantifier(ABC):
    """
        Base class.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def energy(self, x):
        pass


# ---------------
class DenoisingAutoEncoder(nn.Module, Quantifier):
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
            - activation: type of activation function
            - noise: noise amplitude

    """

    def __init__(self, d_input, d_hidden, activation='sigmoid', noise=0):
        super().__init__()
        self.D = d_input
        self.H = d_hidden
        self.noise = noise
        self.activation = activation
        self.encoder = nn.Linear(d_input, d_hidden, bias=False)
        self.bias_h = torch.nn.Parameter(torch.zeros(d_hidden).float())
        self.bias_r = torch.nn.Parameter(torch.zeros(d_input).float())

    # -------
    def energy(self, x):
        if self.activation == 'relu':
            return self.energy_relu(x)
        elif self.activation == 'tanh':
            return self.energy_tanh(x)
        else:
            return self.energy_sigmoid(x) 

    # -------
    def energy_sigmoid(self, x):
        energy = 0
        W = self.encoder.weight
        for i in range(self.H):
            e_unit = torch.exp(torch.matmul(x, W.t()[:,i]) + self.bias_h[i])
            energy -= torch.log(1 + e_unit)
        energy += 0.5 * (x - self.bias_r).norm(p=2).pow(2)
        return energy

    # -------
    def energy_tanh(self, x):
        energy = 0
        W = self.encoder.weight
        return energy

    # -------
    def energy_relu(self, x):
        energy = 0
        W = self.encoder.weight
        return energy

    # -------
    def reconstruction(self, x, add_noise=True):
        return (self.forward(x, add_noise)-x).norm(p=2).pow(2)

    # -------
    def corruption(self, x):
        if self.noise == 0:
            return x
        return x + Variable(x.data.new(x.size()).normal_(0, self.noise))

    # -------
    def forward(self, x, add_noise):
        assert x.size(1) == self.D
        if add_noise:
            x = self.corruption(x)  # N x D
        if self.activation == 'relu':
            u = torch.relu(self.encoder(x) + self.bias_h)  # N x H
        elif self.activation == 'tanh':
            u = torch.tanh(self.encoder(x) + self.bias_h)  # N x H
        else:
            u = torch.sigmoid(self.encoder(x) + self.bias_h)  # N x H
        output = F.linear(u, self.encoder.weight.t()) + self.bias_r  # N x D
        return output


# ---------------
class ConvDenoisingAutoEncoder(nn.Module, Quantifier):
    """
        Convolutional Denoising Autoencoder.
        
        Inputs
            - x: input tensor, size N x 1 x D x D
                    + N: batch size
                    + D: image width/height
            - add_noise: boolean value, true forces the model to use the
                         corruption process

        Arguments
            - d_input: image width/height (D)
            - activation: type of activation function
            - noise: noise amplitude
    """

    def __init__(self, d_input, activation='sigmoid', noise=0):
        super().__init__()
        self.D = d_input
        self.noise = noise
        self.conv1 = nn.Conv2d(1, 16, 3, stride=1, padding=1) 
        self.conv2 = nn.Conv2d(16, 8, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(8, 8, 3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(2, padding=0)
        self.pool2 = nn.MaxPool2d(2, padding=1)

        self.up1 = nn.ConvTranspose2d(8, 8, 3, stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(8, 8, 3, stride=2, padding=1)
        self.up3 = nn.ConvTranspose2d(8, 16, 3, stride=2, padding=0)
        self.conv = nn.Conv2d(16, 1, 4, stride=1, padding=2)

    # -------
    def encoder(self, x):
        conv1 = self.conv1(x)
        relu1 = F.relu(conv1) 
        pool1 = self.pool1(relu1) 
        conv2 = self.conv2(pool1) 
        relu2 = F.relu(conv2)
        pool2 = self.pool1(relu2) 
        conv3 = self.conv3(pool2) 
        relu3 = F.relu(conv3)
        pool3 = self.pool2(relu3) 
        pool3 = pool3.view([x.size(0), 8, 4, 4]).cuda()
        return pool3

    # -------
    def decoder(self, encoding):
        up1 = self.up1(encoding)
        up_relu1 = F.relu(up1) 
        up2 = self.up2(up_relu1) 
        up_relu2 = F.relu(up2)
        up3 = self.up3(up_relu2) 
        up_relu3 = F.relu(up3)
        logits = self.conv(up_relu3)
        logits = F.sigmoid(logits)
        logits = logits.view([encoding.size(0), 1, 28, 28]).cuda()
        return logits

    # -------
    def reconstruction(self, x, add_noise=True):
        return (self.forward(x, add_noise)-x).norm(p=2)

    # -------
    def corruption(self, x):
        return x + Variable(x.data.new(x.size()).normal_(0, self.noise))

    # -------
    def forward(self, x, add_noise):
        assert x.size(2) == self.D and x.size(3) == self.D
        if add_noise:
            x = self.corruption(x)  # N x 1 x D x D
        encoding = self.encoder(x)  # N x ...
        output = self.decoder(encoding)  # N x 1 x D x D
        return encoding, output


        

    







