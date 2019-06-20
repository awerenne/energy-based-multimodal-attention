""" 
    ...
"""

import numpy as np
import matplotlib.pyplot as plt

N = 10000
param_mesh = np.linspace(0, 2*3.1415, N)


# ---------------
def make_wave(n_samples):
    samples = param_mesh[np.random.randint(0, N, n_samples)]
    x1 = samples 
    x2 = np.sin(samples)
    return np.concatenate((np.expand_dims(x1,1), np.expand_dims(x2,1)), axis=1)


# ---------------
def make_circle(n_samples):
    samples = param_mesh[np.random.randint(0, N, n_samples)]
    x1 = np.sin(samples)
    x2 = np.cos(samples)
    return np.concatenate((np.expand_dims(x1,1), np.expand_dims(x2,1)), axis=1)


# ---------------
def make_spiral(n_samples):
    samples = param_mesh[np.random.randint(0, N, n_samples)]
    r = np.sqrt(samples)
    x1 = r * np.cos(samples)
    x2 = r * np.sin(samples)
    return np.concatenate((np.expand_dims(x1,1), np.expand_dims(x2,1)), axis=1)


# ---------------
if __name__ == "__main__":
    # X = make_wave(200)
    # X = make_circle(200)
    X = make_spiral(200)

    plt.scatter(X[:,0], X[:,1], marker='o', alpha=0.3)
    plt.show()





































