"""
    ...
"""

import numpy as np
import torch
import math
import numpy.random
import scipy.stats as ss
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.misc import derivative

numpy.random.seed(0x5eed)

class Mode():
    """
    ...
    """

    def __init__(self, params, e_constant):
        self.params = params
        self.e_constant = e_constant
        self.n_mixtures = params.shape[0] 
        self.make_mesh()
        self.make_abnormal()

    def make_mesh(self):
        samples = self.sample(10000)
        self.minx = samples.min()
        self.maxx = samples.max()
        self.xmesh = np.linspace(self.minx-5, self.maxx+5, 3000)

    def make_abnormal(self):
        self.laplace = derivative(self.likelihood, self.xmesh, dx=1e-6, n=2)
        self.laplace /= self.likelihood(self.xmesh)
        self.laplace -= 6
        temp = self.laplace * np.roll(self.laplace, 1)
        temp = np.where(temp < 0)[0]
        self.intervals = []
        for i in range(len(temp)-1):
            a, b = float(temp[i]), float(temp[i+1])
            mid = int(a + math.ceil((b-a)/2))
            if self.laplace[mid] < 0: 
                self.intervals.append((self.xmesh[temp[i]],
                        self.xmesh[temp[i+1]]))
        return self

    def get_laplacian(self):
        return self.laplace  

    def A(self, x):
        for a,b in self.intervals:
            if x >= a and x <= b: return 1./self.n_mixtures
        return 0

    def likelihood(self, x):
        p = np.zeros_like(x)
        for mean, stddev in self.params:
            p += ss.norm.pdf(x, loc=mean, scale=stddev) * 1/self.n_mixtures  
        return p

    def energy(self, x):
        p = self.likelihood(x)
        return -np.log(p+1e-9) + self.e_constant

    def laplacian(self, x):
        pass

    def plot_pdf(self):
        ymesh = self.likelihood(self.xmesh)
        x = self.sample() 
        plt.plot(self.xmesh, ymesh)
        plt.scatter(x, np.zeros_like(x))
        plt.xlabel("x")
        plt.show()

    def plot_laplacian(self):
        a = np.asarray(list(map(self.A, list(self.xmesh))))
        normal = np.where(a > 0)[0]
        ymesh = self.energy(self.xmesh)
        plt.plot(self.xmesh, ymesh)
        plt.scatter(self.xmesh[normal], ymesh[normal], c='r', marker='*')
        plt.xlabel("x")
        plt.show()

    def plot_energy(self):
        ymesh = self.energy(self.xmesh)
        x = self.sample() 
        y = self.energy(x)
        a = np.asarray(list(map(self.A, list(x))))
        normal = a[a > 0]
        plt.plot(self.xmesh, ymesh)
        plt.scatter(x[a>0], y[a>0], facecolors='none', edgecolors='r')
        plt.scatter(x[a<=0], y[a<=0], facecolors='none', edgecolors='m')
        plt.xlabel("x")
        plt.ylabel("phi(x)")
        plt.show()

    def sample(self, n_samples=50):
        mixture_idx = np.random.choice(self.n_mixtures, size=n_samples)
        samples = np.fromiter((ss.norm.rvs(*(self.params[i])) for i in \
                mixture_idx), dtype=np.float64)
        return samples

    def forced_sample(self, n_samples, frac=0.25):
        n_abnormal = int(frac*n_samples)
        n_normal = n_samples - n_abnormal
        x_normal = np.zeros((n_normal,3))
        idx_normal = 0
        for i in range(n_normal):
            mixture_idx = np.random.choice(self.n_mixtures, size=n_normal)
            x = np.fromiter((ss.norm.rvs(*(self.params[i])) for i in \
                    mixture_idx), dtype=np.float64)
            for j in range(len(x)):
                if self.A(x[j]) > 0:
                    n_normal -= 1
                    x_normal[idx_normal,0] = x[j]
                    x_normal[idx_normal,1] = self.A(x[j])
                    x_normal[idx_normal,2] = self.energy(x[j])
                    idx_normal += 1
                    if idx_normal == len(x_normal):
                        n_normal = 0
                        break

        x_abnormal = np.zeros((n_abnormal,3))
        idx_abnormal = 0
        for i in range(n_abnormal):
            x = np.random.choice(self.xmesh, size=n_abnormal)
            for j in range(len(x)):
                if self.A(x[j]) == 0:
                    n_abnormal -= 1
                    x_abnormal[idx_abnormal,0] = x[j]
                    x_abnormal[idx_abnormal,1] = self.A(x[j])
                    x_abnormal[idx_abnormal,2] = self.energy(x[j])
                    idx_abnormal += 1
                    if idx_abnormal == len(x_abnormal):
                        n_abnormal = 0
                        break
        x = np.concatenate((x_normal,x_abnormal),axis=0)
        return x

# params = np.array([[12, 1.2], [2, 0.7]])
# params = np.array([[4, 1.6]])
# params = np.array([[4, 0.4]])
# m = Mode(params, 5)
# m.plot_pdf()
# m.plot_energy()
# m.plot_laplacian()
# x = m.forced_sample(100)
# plt.scatter(a[:,0], np.zeros_like(a[:,0]), facecolors='none', edgecolors='r')
# plt.scatter(b[:,0], np.zeros_like(b[:,0]), facecolors='none', edgecolors='m')
# plt.xlabel("x")
# plt.ylabel("phi(x)")
# plt.show()


    




























