import torch


def softargmax(x, beta = 1, n_out = 1, support = None):
    if support is None:
        support = Variable(torch.from_numpy(np.arange(1,x.shape[-1]+1))).float()
    return torch.sum(support*torch.exp(beta*x)/torch.sum(torch.exp(beta*x),-1)[:,:,None],-1)
