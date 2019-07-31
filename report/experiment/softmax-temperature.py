import matplotlib.pyplot as plt
import torch
import seaborn as sns
from matplotlib import rc
import numpy as np
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
plt.style.use('seaborn-whitegrid')
width = 0.15 

x = np.arange(2)
energies = torch.tensor([10, 30]).float()
plt.bar(x, energies.data.numpy(), width=width)
plt.ylabel('')
plt.xticks(x, (r'$E_1$', r'$E_2$'))
plt.grid(False)
plt.ylim((0,50))
plt.ylabel('Energy', fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=16)
# plt.savefig('input-gibbs.pdf')
plt.show()


# scores_high_temp = torch.softmax(-1./1000 * energies.unsqueeze(-1), dim=0)
# scores_low_temp = torch.softmax(-1./10 * energies.unsqueeze(-1), dim=0)
# N = 2
# idx = np.arange(N)     
# plt.bar(idx, scores_low_temp.squeeze().data.numpy(), width, label='Low temperature')
# plt.bar(idx + width, scores_high_temp.squeeze().data.numpy(), width, label='High temperature')
# plt.xticks(idx + width / 2, (r'$\alpha_1$', r'$\alpha_2$'))
# plt.grid(False)
# plt.legend(loc='best', prop={'size': 13})
# plt.ylabel('Output Gibbs', fontsize=18)
# plt.tick_params(axis='both', which='major', labelsize=16)
# plt.savefig('result-gibbs.pdf')
# plt.show()