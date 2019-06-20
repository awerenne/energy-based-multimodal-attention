import numpy as np
import numpy.random
import scipy.stats as ss
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils import data

from data import Mode
from model import Composer


# m1 = Mode(np.array([[4, 1.2]]), 2)
# m1 = Mode(np.array([[12, 1.2], [2, 0.7]]), 0.5)
m1 = Mode(np.array([[4, 1.6]]), 100)
m2 = Mode(np.array([[4, 1.6]]), 1)
# m2 = Mode(np.array([[4, 0.4]]), 1)
model = Composer(2).float()
model.reset(200)
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
optimizer = torch.optim.Adagrad(model.parameters())

def regul(logits, abn1, abn2):
    loss = (logits[:,0] * abn1) + (logits[:,1] * abn2)
    return -torch.sum(loss)

X = torch.zeros(1000,3,2)
temp = m1.forced_sample(1000, frac=0.75)
np.random.shuffle(temp)
X[:,:,0] = torch.tensor(temp)
temp = m2.forced_sample(1000)
np.random.shuffle(temp)
X[:,:,1] = torch.tensor(temp)
X = X.float()

print(dict(model.named_parameters())['Gamma'])
print(dict(model.named_parameters())['w'])
print(model.mode_min)

n_epochs = 100
batch_size = 8
for i in range(n_epochs):
    indices = np.arange(X.size(0))
    np.random.shuffle(indices)
    for j in range(0, len(X)-batch_size, batch_size):
        optimizer.zero_grad()
        idx = indices[j:j+batch_size]
        alphas, logits = model(X[idx,2,:].squeeze())
        loss = regul(logits, X[idx,1,0], X[idx,1,1])
        if j%20 == 0: 
            print(loss.item())
        loss.backward()
        optimizer.step()
        dict(model.named_parameters())['Gamma'].data.clamp_(min=1e-5, max=1-1e-5)
        dict(model.named_parameters())['w'].data.clamp_(1e-5)
        model.cooldown()


print(dict(model.named_parameters())['Gamma'])
print(dict(model.named_parameters())['w'])
print(model.mode_min)


import numpy as np
import numpy.random
import scipy.stats as ss
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils import data

from data import Mode
from model import Composer


m1 = Mode(np.array([[4, 1.]]), 5)
m2 = Mode(np.array([[4, 1.]]), 5)
m3 = Mode(np.array([[4, 1.]]), 5)
model = Composer(3).float()
model.reset(500)
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
optimizer = torch.optim.Adagrad(model.parameters())

def regul(logits, abn):
    loss = 0
    for i in range(logits.size(-1)):
        loss += logits[:,i] * abn[:,i]
    cost = -torch.sum(loss)
    return cost

X = torch.zeros(1000,3,3)
X[:,:,0] = torch.tensor(m1.forced_sample(1000))

# X[:,:,1] = torch.tensor(m2.forced_sample(1000))
X[:,:,1] = X[:,:,0]
X[:,2,1] = X[:,2,1] * X[:,2,1]

temp = m3.forced_sample(1000)
# print(temp[:3])
np.random.shuffle(temp)
# print(temp[:3])
X[:,:,2] = torch.tensor(temp)
X = X.float()

print("\nGamma")
print(dict(model.named_parameters())['Gamma'])
print("\nw")
print(dict(model.named_parameters())['w'])
print("\ncorrection")
print(model.mode_min)

n_epochs = 100
batch_size = 16
for i in range(n_epochs):
    indices = np.arange(X.size(0))
    np.random.shuffle(indices)
    for j in range(0, len(X)-batch_size, batch_size):
        optimizer.zero_grad()
        idx = indices[j:j+batch_size]
        alphas, logits = model(X[idx,2,:].squeeze())
        loss = regul(logits, X[idx,1,:].squeeze())
        if j%20 == 0: 
            print(loss.item())
        loss.backward()
        optimizer.step()
        dict(model.named_parameters())['Gamma'].data.clamp_(min=1e-5, max=1-1e-5)
        dict(model.named_parameters())['w'].data.clamp_(1e-5)
        model.cooldown()


print("\nGamma")
print(dict(model.named_parameters())['Gamma'])
print("\nw")
print(dict(model.named_parameters())['w'])
print("\ncorrection")
print(model.mode_min)











































