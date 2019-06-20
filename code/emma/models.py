"""
    ...
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


#---------------
class Encoder(nn.Module):
    """
        ...
    """

    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 512)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        m = F.relu(self.fc1(x))
        return m


#---------------
class SingleDecoder(nn.Module):
    def __init__(self):
        super(SingleDecoder, self).__init__()
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, m):
        m = F.relu(self.fc1(m))
        m = self.fc2(m)
        return F.log_softmax(m, dim=-1)


#---------------
class DualDecoder(nn.Module):
    def __init__(self):
        super(DualDecoder, self).__init__()
        self.fc1 = nn.Linear(1024, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, m):
        m = F.relu(self.fc1(m))
        m = self.fc2(m)
        return F.log_softmax(m, dim=-1)











    







