import matplotlib.pyplot as plt
import torch
import seaborn as sns
from matplotlib import rc
import numpy as np
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
plt.style.use('seaborn-whitegrid')

x = np.arange(2)
energies = torch.tensor([10, 30]).float()
plt.bar(x, energies.data.numpy(), width=0.35)
plt.ylabel('Energy')
plt.xticks(x, (0, 1))
plt.grid(False)
plt.ylim((0,50))
plt.savefig('input-gibbs.pdf')
plt.show()
scores_high_temp = torch.softmax(-1./1000 * energies.unsqueeze(-1), dim=0)
# plt.bar(x, scores_high_temp.squeeze().data.numpy())
# plt.xticks(x, (0, 1))
# plt.grid(False)
# plt.ylim((0,1))
# plt.show()
scores_low_temp = torch.softmax(-1./10 * energies.unsqueeze(-1), dim=0)
# plt.bar(x, scores_high_temp.squeeze().data.numpy(), width=0.3, align='center')
# plt.bar(x, scores_low_temp.squeeze().data.numpy(), width=0.3, align='center')
# plt.xticks(x+0.15, (0, 1))
# plt.grid(False)
# plt.ylim((0,1))
# plt.show()

N = 2
idx = np.arange(N) 
width = 0.35       
plt.bar(idx, scores_low_temp.squeeze().data.numpy(), width, label='Low temperature')
plt.bar(idx + width, scores_high_temp.squeeze().data.numpy(), width, label='High temperature')

plt.ylabel('Output Gibbs')
# plt.title('Scores by group and gender')
plt.xticks(idx + width / 2, ('0', '1'))
plt.grid(False)
plt.legend(loc='best')
plt.savefig('result-gibbs.pdf')
plt.show()