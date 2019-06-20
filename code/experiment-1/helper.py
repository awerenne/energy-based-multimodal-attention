import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d.axes3d import Axes3D


# ---------------
def plot_curves(curves):
    train_curve, test_curve = curves
    train_curve = np.asarray(train_curve)
    test_curve = np.asarray(test_curve)
    epochs = np.arange(len(train_curve))
    plt.plot(epochs, train_curve)
    plt.plot(epochs, test_curve)
    plt.show()

# ---------------
def make_manifold(n_samples, type_manifold):
    if type_manifold == 1:
        X = make_linear(n_samples, 2)
    elif type_manifold == 2:
        X = make_curved(n_samples, 2)
    elif type_manifold == 3:
        X = make_circle(n_samples, 2)
    else:
        X = make_spiral(n_samples, 2)
    return X
    

# ---------------
def make_loaders(X):
    X_train, X_test = train_test_split(X, test_size=0.33, random_state=seed)
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train).float())
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=16)
    test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test).float())
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=16)
    return (train_loader, test_loader)
    
# ---------------
def plot_data(X):
    data = X.data.numpy()
    plt.scatter(data[:, 0], data[:, 1], c='blue', edgecolor='k')
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.show()


# ---------------
def make_mesh():
    # nx, ny = (40, 40)
    # x = np.linspace(-2, 8, nx)
    # y = np.linspace(-2, 8, ny)
    # nx, ny = (60, 60)
    nx, ny = (30, 30)
    x = np.linspace(-6, 12, nx)
    y = np.linspace(-6, 12, ny)
    x = np.linspace(-1, 3, nx)
    y = np.linspace(-1, 3, ny)
    # x = np.linspace(-80, 80, nx)
    # y = np.linspace(-80, 80, ny)
    # x = np.linspace(0, 2, nx)
    # y = np.linspace(0, 2, ny)
    xv, yv = np.meshgrid(x, y)
    return xv, yv, nx, ny


# ---------------
def plot_vector_field(X, model, save=False):
    xv, yv, nx, ny = make_mesh()
    U = np.zeros(xv.shape)
    V = np.zeros(yv.shape)
    for i in range(nx):
        for j in range(ny):
            x = xv[i,j]
            y = yv[i,j]
            sample = torch.tensor([x, y]).float()
            sample = sample.unsqueeze(0)
            r = model(sample)
            r = r.squeeze()
            U[i,j] = r[0] - x
            V[i,j] = r[1] - y

    fig1, ax1 = plt.subplots()
    ax1.set_title('...')
    norms = np.sqrt(U**2 + V**2)
    import matplotlib.colors as colors
    Q = ax1.quiver(xv, yv, U, V)

    data = np.zeros((100,2))
    k = 0
    for i in range(len(X)): 
        x1 = X[i,0]
        x2 = X[i,1]
        # if x1 <= 12 and x1 >= -3.5 and x2 <= 12 and x2 >= -3.5:
        if x1 <= 3 and x1 >= -1 and x2 <= 3 and x2 >= -1:
            data[k,0] = x1
            data[k,1] = x2
            k += 1
            if k > len(data)-1: break
    plt.scatter(data[:, 0], data[:, 1], c='blue', edgecolor='k')
    if save: plt.savefig('results/vfield')
    else: plt.show()


# ---------------
def plot_energy(model, save=False):
    xv, yv, nx, ny = make_mesh()
    Z = np.zeros(yv.shape)
    Vmin = float("inf")
    for i in range(nx):
        for j in range(ny):
            x = xv[i,j]
            y = yv[i,j]
            sample = torch.tensor([x, y]).float()
            sample = sample.unsqueeze(0)
            energy = model.energy(sample)
            if energy < Vmin: Vmin = energy
    for i in range(nx):
        for j in range(ny):
            x = xv[i,j]
            y = yv[i,j]
            sample = torch.tensor([x, y]).float()
            sample = sample.unsqueeze(0)
            energy = model.energy(sample) - Vmin
            # Z[i,j] = energy.data
            Z[i,j] = torch.log(1e-1 + energy.data)
            # Z[i,j] = torch.log(1e-1 + (model(sample)-sample).norm(p=2))
    plt.pcolormesh(xv, yv, Z, cmap = 'plasma') 
    plt.colorbar()
    if save: plt.savefig('results/energy')
    else: plt.show()


# ---------------
def plot_laplacian(model):
    xv, yv, nx, ny = make_mesh()
    Z = np.zeros(yv.shape)
    # X_noisy = torch.zeros(nx,2).float()
    # for j in range(2):
    #     n = 0.3 * np.random.normal(loc=0.0, scale=1,
    #             size=nx)
    #     X_noisy[:,j] += torch.tensor(n).float()
    for i in range(nx):
        for j in range(ny):
            x = xv[i,j]
            y = yv[i,j]
            # sample = torch.tensor([x+X_noisy[i,0], y+X_noisy[j,1]]).float()
            sample = torch.tensor([x, y]).float()
            sample = sample.unsqueeze(0)
            laplacian = model.laplacian(sample)
            Z[i,j] = laplacian
    plt.pcolormesh(xv, yv, Z, cmap = 'plasma') 
    plt.colorbar()
    plt.show()



    

    




























