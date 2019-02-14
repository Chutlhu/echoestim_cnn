import numpy
import pickle as pkl
import matplotlib.pylab as plt

filename = './results/performance.pkl'
with open(filename,'rb') as file:
    res = pkl.load(file)

print(res.keys())

for phase in ['train', 'valid', 'overfit']:
    plt.plot(res['train']['error'])
    plt.plot(res['train']['err_tdoa'])
    plt.plot(res['train']['err_itdoa'])
    plt.plot(res['train']['err_tdoe'])
    plt.title(phase)
    plt.legend(['all', 'tdoa', 'itdoa', 'tdoe'])
    plt.show()

print(res['test'])
