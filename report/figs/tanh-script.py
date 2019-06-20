import matplotlib.pyplot as plt 
import numpy as np 
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
plt.style.use('seaborn-whitegrid')

def relu(f):
    y = []
    for x in f:
        if x < 0: y.append(0)
        else: y.append(x)
    return np.asarray(y)

alpha = np.arange(-1,2,0.01)
G=0.9
b=0.8
sigma = relu(np.tanh(G*alpha-b))
plt.plot(alpha, sigma, label='weak')
G=2
b=0.1
sigma = relu(np.tanh(G*alpha-b))
plt.plot(alpha, sigma, label='strong')
plt.legend(loc='upper left', fontsize=18)
plt.xlim([-1,2])
plt.ylim([-0.5,1.5])
plt.xlabel(r'$\alpha$', fontsize=18)
plt.ylabel(r'$\beta$', fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=11)
plt.savefig('tanh.png')
plt.show()






