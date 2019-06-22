import os
from torch.utils import data
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 


#---------------
# def make_data(name="training"):
#     X, y = torch.load('./data_mnist/processed/' + name + '.pt')
        
#     numpy_y = y.data.numpy()
#     sorted_y = np.sort(numpy_y)
#     idx_X = np.argsort(numpy_y)
#     _, occur_y = np.unique(numpy_y, return_counts=True)
#     n_zeros = occur_y[0]
#     n_train = int(0.7*float(n_zeros))
#     n_valid = n_zeros - n_train
#     X0_train = torch.zeros(n_train, X.size(1), X.size(2))
#     X0_valid = torch.zeros(n_valid, X.size(1), X.size(2))
#     X1_valid = torch.zeros(n_valid, X.size(1), X.size(2))

#     X0_train = X[idx_X[0:0+n_train]]
#     X0_valid = X[idx_X[n_train:n_train+n_valid]]
#     X1_valid = X[idx_X[occur_y[1]:occur_y[1]+n_valid]]

#     torch.save([X0_train, X0_valid, X1_valid], 'test_data/zeroandone.pt')


#---------------
def Generator():
    for chunk in pd.read_csv("../datasets/higgs/HIGGS.csv", header=None, chunksize=1e5):
        X1 = torch.tensor(chunk[range(1,22)].values)
        X2 = torch.tensor(chunk[range(22,29)].values)
        X = [X1, X2]
        y = torch.tensor(chunk[0].values)
        yield (X, y)
















