import numpy as np
import matplotlib.pyplot as plt


radius_noise = 0
N = 10000
param_mesh = np.linspace(0, 5, N)
# radius_noise = 1e-1


# ---------------
def make_linear(n_samples, bias):
    samples = param_mesh[np.random.randint(0, N, n_samples)]
    w = 2
    x1 = samples + bias
    x2 = w * samples + bias
    # x1 += np.random.normal(0, radius_noise, n_samples)
    # x2 += np.random.normal(0, radius_noise, n_samples)
    return np.concatenate((np.expand_dims(x1,1),np.expand_dims(x2,1)), axis=1)


# ---------------
def make_curved(n_samples, bias):
    samples = param_mesh[np.random.randint(0, N, n_samples)]
    omega = 5/3.14
    x1 = samples + bias
    x2 = np.sin(omega * samples)
    x1 += np.random.normal(0, radius_noise, n_samples)
    x2 += np.random.normal(0, radius_noise, n_samples)
    return np.concatenate((np.expand_dims(x1,1),np.expand_dims(x2,1)), axis=1)


# ---------------
def make_circle(n_samples, bias):
    samples = param_mesh[np.random.randint(0, N, n_samples)]
    x1 = np.sin(samples) + bias
    x2 = np.cos(samples) + bias
    x1 += np.random.normal(0, radius_noise, n_samples)
    x2 += np.random.normal(0, radius_noise, n_samples)
    return np.concatenate((np.expand_dims(x1,1),np.expand_dims(x2,1)), axis=1)


# ---------------
def make_spiral(n_samples, bias):
    samples = param_mesh[np.random.randint(0, N, n_samples)]
    r = np.sqrt(samples)
    x1 = r * np.cos(samples) + bias
    x2 = r * np.sin(samples) + bias
    x1 += np.random.normal(0, radius_noise, n_samples)
    x2 += np.random.normal(0, radius_noise, n_samples)
    return np.concatenate((np.expand_dims(x1,1),np.expand_dims(x2,1)), axis=1)



# ---------------
if __name__ == "__main__":
    # X = make_circle(200, 2)
    X = make_spiral(200, 2)
    plt.scatter(X[:,0], X[:,1])
    plt.show()





































