import os
from torch.utils import data
import torch
import numpy as np
import matplotlib.pyplot as plt


#---------------
def make_data(name="training"):
    Xm, ym = torch.load('./data_mnist/processed/' + name + '.pt')
    Xf, yf = torch.load('./data_fashion/processed/' + name + '.pt')
        
    numpy_ym = ym.data.numpy()
    sorted_ym = np.sort(numpy_ym)
    idx_Xm = np.argsort(numpy_ym)
    _, occur_ym = np.unique(numpy_ym, return_counts=True)

    numpy_yf = yf.data.numpy()
    sorted_yf = np.sort(numpy_yf)
    idx_Xf = np.argsort(numpy_yf)
    _, occur_yf = np.unique(numpy_yf, return_counts=True)

    sizes = []
    for n1, n2 in zip(occur_ym, occur_yf):
        sizes.append(np.min([n1,n2]))

    N = np.sum(sizes)
    X_new = torch.zeros(N, 2, Xm.size(1), Xm.size(2))
    y_new = torch.zeros(N)

    begin_new = 0
    begin_old_f = 0
    begin_old_m = 0
    for i in range(10):
        size = sizes[i]
        y_new[begin_new:begin_new+size] = torch.tensor(sorted_yf[begin_old_f:begin_old_f+size])
        
        idx = idx_Xm[begin_old_m:begin_old_m+size]
        X_new[begin_new:begin_new+size,0] = Xm[idx]

        idx = idx_Xf[begin_old_f:begin_old_f+size]
        X_new[begin_new:begin_new+size,1] = Xf[idx]
        
        begin_new += size
        begin_old_f += occur_yf[i]
        begin_old_m += occur_ym[i]
    torch.save([X_new, y_new], 'data_combine/' + name + '.pt')


#---------------
class CombineDataset(data.Dataset):
    """
        ...
    """

    def __init__(self, train=True, transform=None):
        if train: file = 'training.pt'
        else: file = 'test.pt'
        self.data, self.targets = torch.load(os.path.join('data_combine/', file))
        self.transform = transform

    def __len__(self):
        return self.targets.size(0)

    def __getitem__(self, index):
        x = self.data[index]
        if self.transform is not None:
            x = self.transform(x)
        return x, self.targets[index]


# make_data("training")
# make_data("test")
# Dataset()




















