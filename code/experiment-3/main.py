"""
    ...
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from data import CombineDataset
from models import Encoder, DualDecoder


#---------------
def forward(model, data):
    m1 = model[0](data[:,0].unsqueeze(1))
    m2 = model[1](data[:,1].unsqueeze(1))
    m = torch.cat((m1, m2), dim=-1)
    return model[-1](m)


#---------------
def train(model, train_loader, optimizer, epoch):
    model[-1].train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = forward(model, data)
        loss = F.nll_loss(output, target.long())
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


#---------------
def test(model, test_loader):
    model[-1].eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = forward(model, data)
            test_loss += F.nll_loss(output, target.long(), reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.long().view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


#---------------       
if __name__ == '__main__':
    
    train_loader = torch.utils.data.DataLoader(CombineDataset(train=True,
        transform=transforms.Normalize((0.1307,), (0.3081,))),
        batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(CombineDataset(train=False,
        transform=transforms.Normalize((0.1307,), (0.3081,))),
        batch_size=32, shuffle=True)

    enc1 = Encoder().float()
    enc1.load_state_dict(torch.load("models/mnist_encoder.pt"))
    enc2 = Encoder().float().eval()
    enc2.load_state_dict(torch.load("models/fashion_encoder.pt"))
    dec = DualDecoder().float().eval()
    optimizer = optim.SGD(dec.parameters(), lr=0.01, momentum=0.9)
    model = nn.ModuleList([enc1, enc2, dec])

    n_epochs = 1
    for epoch in range(1, n_epochs+1):
        train(model, train_loader, optimizer, epoch)
        torch.save(model[-1].state_dict(),"models/dualdecoder.pt")
        test(model, test_loader)
    
#---------------
#---------------
#---------------
#---------------


from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from models import Encoder, SingleDecoder

def train(model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        print(data.size())
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

        
if __name__ == '__main__':

    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('./data_fashion', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('./data_fashion', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=32, shuffle=True)

    model = nn.Sequential(Encoder(), SingleDecoder())
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    n_epochs = 1
    for epoch in range(1, 1+n_epochs):
        train(model, train_loader, optimizer, epoch)
        test(model, test_loader)
    # torch.save(model[0].state_dict(),"models/fashion_encoder.pt")


#---------------
#---------------
#---------------
#---------------





from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from models import Encoder, SingleDecoder

    
def train(model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

        
if __name__ == '__main__':

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data_mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data_mnist', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=32, shuffle=True)

    model = nn.Sequential(Encoder(), SingleDecoder())
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    n_epochs = 1
    for epoch in range(1, 1+n_epochs):
        train(model, train_loader, optimizer, epoch)
        test(model, test_loader)
    # torch.save(model[0].state_dict(),"models/mnist_encoder.pt")



#---------------
#---------------
#---------------
#---------------
#---------------



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











    


























