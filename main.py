import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.autograd import Variable

## PERFORMANCE METRICS
def nrmse(x, x_ref):
    return np.sqrt(np.sum((x - x_ref)**2, axis = 0))\
           /np.sqrt(np.sum((x_ref - np.mean(x_ref, axis = 0))**2, axis = 0))

filename_vast = 'mirage_task10_train_vars.mat'
filename_rec  = 'mirage_rec_task00_train.mat'

# HYPER-PARAMS
net_type = 'girin'
n_test   = 200  # dimension of test set
n_valid  = 500
n_epochs = 200  # or whatever
batch_size = 1000 # or whatever
patience = 30

## IMPORT DATA
print('Loading data...')

import h5py
if net_type is 'habets':
    with h5py.File('data/' + filename_rec, 'r') as file:
        data_rec = np.array(file['rec_dataset'])
        n_sig, n_chan, n_rirs = data_rec.shape
        # FFT params
        print('Organizing features...')
        n_fft = 513
        data_rec_fft = np.zeros([n_rirs, n_chan, 2*n_fft])
        from scipy import signal

        for i in range(n_rirs):
            for j in range(n_chan):
                _, _, Zxx = signal.stft(data_rec[:,j,i], nfft = 2*(n_fft-1))
                A = np.mean(np.abs(Zxx), axis = 1)
                p = np.mean(np.angle(Zxx), axis = 1)
                data_rec_fft[i,j,:] = np.concatenate([A, p])
        print(n_sig, n_chan, n_rirs, N)

data_vast = scipy.io.loadmat('data/' + filename_vast)
print('done.\n')

tdoa = data_vast['tdoa']
itdoa = data_vast['itdoa']
tdoe = data_vast['tau12'] - data_vast['tau11']

N,_ = tdoa.shape

ILD  = data_vast['ILD']
iIPD = data_vast['iIPD']
rIPD = data_vast['rIPD']

del data_vast
if net_type is 'habets':
    del data_rec

iIPD = np.hstack([np.zeros([N,1]),iIPD]) # add an empty value for concatenation
rIPD = np.hstack([np.zeros([N,1]),rIPD]) # add an empty value for concatenation
print('done.\n')

print('Training and Test set...')
# var = 1e4*tdoa   # N samples
fs = 16000
var = np.stack([fs*tdoa, fs*itdoa, fs*tdoe], axis = 1).squeeze() # Nx3 matrix

if net_type is 'habets':
    obs = data_rec_fft          # FxN matrix
if net_type is 'girin':
    obs = np.stack([ILD, rIPD,iIPD], axis=2) # Fx3xN matrices

n_obs,n_feat,n_dims = obs.shape
n_train = n_obs - n_test

# Training set
random_indeces = np.random.permutation(n_obs)
train_var = var[random_indeces[0:n_train-n_valid],:]
train_obs = obs[random_indeces[0:n_train-n_valid],:]
# Overfitting set
random_subindeces = np.random.permutation(n_train-n_valid)
overf_var = var[random_indeces[random_subindeces[0:200]],:]
overf_obs = obs[random_indeces[random_subindeces[0:200]],:]
# Validation set
valid_var = var[random_indeces[n_train-n_valid:n_train],:]
valid_obs = obs[random_indeces[n_train-n_valid:n_train],:]
# Test set
test_var = var[random_indeces[n_train:n_obs],:]
test_obs = obs[random_indeces[n_train:n_obs],:]


train_t = Variable(torch.from_numpy(train_var)).float()
train_y = Variable(torch.from_numpy(train_obs[:,None,:,:])).float()

overf_t = Variable(torch.from_numpy(overf_var)).float()
overf_y = Variable(torch.from_numpy(overf_obs[:,None,:,:])).float()

valid_t = Variable(torch.from_numpy(valid_var)).float()
valid_y = Variable(torch.from_numpy(valid_obs[:,None,:,:])).float()

test_t = Variable(torch.from_numpy(test_var)).float()
test_y = Variable(torch.from_numpy(test_obs[:,None,:,:])).float()
print('done.\n')

N, C, H, W = train_y.data.numpy().shape # Batch x Channels x Freq x Dim
_, L = train_t.data.numpy().shape

xmin, xmax = np.min(tdoa)-0.0005, np.max(tdoa)+0.0005
ymin, ymax = xmin, xmax

## NETWORK
class ConvNet_Girin(torch.nn.Module):
    def __init__(self):
        super(ConvNet_Girin, self).__init__()

        self.conv1 = torch.nn.Sequential(
            # conv2d: in_channels, out_channels, kernel_size
            # in_channels = 1
            # n_filters = 24
            torch.nn.Conv2d(1, 24, kernel_size=3, stride = 1, padding = 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.BatchNorm2d(24))
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(24, 48, kernel_size=3, stride = 1, padding = 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.BatchNorm2d(48))
        self.drop_out = torch.nn.Dropout()
        self.fc1 = torch.nn.Linear(48*128, 360)
        self.fc2 = torch.nn.Linear(360, 240)
        self.fc3 = torch.nn.Linear(240, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.drop_out(x)
        x = x.view(-1, 48*128)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ConvNet_Habets(torch.nn.Module):
    def __init__(self):
        super(ConvNet_Habets, self).__init__()

        self.conv1 = torch.nn.Sequential(
            # inputs in the form MxF (M mics x F fft points)
            # conv2d: in_channels, out_channels, kernel_size
            # in_channels = 1
            # n_filters = 64
            torch.nn.Conv2d(1, 64, kernel_size=(2,1), stride = 1, padding = 0),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.BatchNorm2d(64))
        self.drop_out = torch.nn.Dropout()
        self.fc1 = torch.nn.Linear(64*513, 512)
        self.fc2 = torch.nn.Linear(512, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.drop_out(x)
        x = x.view(-1, 64*513)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# define the network
if net_type is 'girin':
    model = ConvNet_Girin()
if net_type is 'habets':
    model = ConvNet_Habets()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

plt.ion()

indeces = [(i,i+batch_size) for i in range(N) if i%batch_size==0 and i < N]

performance = { x : { 'error' : [],
                      'loss'  : [],
                      'loss_tdoa'  : [],
                      'loss_itdoa' : [],
                      'loss_tdoe'  : [],
                      'err_tdoa'  : [],
                      'err_itdoa' : [],
                      'err_tdoe'  : [],
                    }
               for x in ['train', 'valid', 'overfit', 'test']}

converged = False

## TRAINING
for it in range(n_epochs):

    if converged:
        break

    for phase in ['train', 'overfit', 'valid']:

        print('Phase', phase)

        if phase == 'train':
            model.train()

            for i, (start, end) in enumerate(indeces):

                prediction = model(train_y[start:end,...])     # input x and predict based on x

                loss1 = loss_func(prediction[:,0], train_t[start:end,0])     # must be (1. nn output, 2. target)
                loss2 = loss_func(prediction[:,1], train_t[start:end,1])     # must be (1. nn output, 2. target)
                loss3 = loss_func(prediction[:,2], train_t[start:end,2])     # must be (1. nn output, 2. target)
                loss = loss1 + loss2 + loss3

                optimizer.zero_grad()   # clear gradients for next train
                loss.backward()         # backpropagation, compute gradients
                optimizer.step()        # apply gradients

                # Track the accuracy
                total = train_t.size(0)
                error = np.mean(nrmse(prediction.data.numpy(), train_t[start:end].data.numpy()))
                error1 = nrmse(prediction.data.numpy()[:,0], train_t[start:end,0].data.numpy())
                error2 = nrmse(prediction.data.numpy()[:,1], train_t[start:end,1].data.numpy())
                error3 = nrmse(prediction.data.numpy()[:,2], train_t[start:end,2].data.numpy())

                if (i) % 10 == 0:
                    print('  Epoch [{}/{}], Batch [{:02d}/{:02d}], Loss: {:.4f}, nRMSE: {:.2f}%' \
                          .format(it + 1, n_epochs, i, len(indeces), loss.item(), error))


        if phase == 'overfit':
            model.eval()
            prediction = model(overf_y)
            error = np.mean(nrmse(prediction.data.numpy(), overf_t.data.numpy()))
            error1 = nrmse(prediction.data.numpy()[:,0], overf_t[:,0].data.numpy())
            error2 = nrmse(prediction.data.numpy()[:,1], overf_t[:,1].data.numpy())
            error3 = nrmse(prediction.data.numpy()[:,2], overf_t[:,2].data.numpy())
            print('  Epoch [{}/{}], nRMSE: {:.2f}, TODA: {:.3f}, iTDOA: {:.3f}, TDOE: {:.3f}%' \
                  .format(it + 1, n_epochs, error, error1, error2, error3))

        if phase == 'valid':
            model.eval()
            prediction = model(valid_y)
            error = np.mean(nrmse(prediction.data.numpy(), valid_t.data.numpy()))
            error1 = nrmse(prediction.data.numpy()[:,0], valid_t[:,0].data.numpy())
            error2 = nrmse(prediction.data.numpy()[:,1], valid_t[:,1].data.numpy())
            error3 = nrmse(prediction.data.numpy()[:,2], valid_t[:,2].data.numpy())
            print('  Epoch [{}/{}], nRMSE: {:.2f}, TODA: {:.3f}, iTDOA: {:.3f}, TDOE: {:.3f}%' \
                  .format(it + 1, n_epochs, error, error1, error2, error3))
            # best model
            if it > 2 and error < np.min(np.array(performance[phase]['error'])):
                best_model = model
                best_performance = performance
            # early stopping
            if it > patience:
                last_error = error
                min_last_patience_errors = np.min(np.array(performance[phase]['error'][-patience:-1]))
                print(np.abs(last_error - min_last_patience_errors))
                if last_error > min_last_patience_errors:
                    converged = True
                print('Early stopping:', error, min_last_patience_errors, ' - Converged?', converged)
            print('\n')

        performance[phase]['error'].append(error)
        performance[phase]['loss'].append(loss.item())
        performance[phase]['err_tdoa'].append(error1)
        performance[phase]['err_itdoa'].append(error2)
        performance[phase]['err_tdoe'].append(error3)
        print('\n')

print('training ends.\n')

phase = 'test'
prediction = model(test_y)
error = np.mean(nrmse(prediction.data.numpy(), test_t.data.numpy()))
error1 = nrmse(prediction.data.numpy()[:,0], test_t[:,0].data.numpy())
error2 = nrmse(prediction.data.numpy()[:,1], test_t[:,1].data.numpy())
error3 = nrmse(prediction.data.numpy()[:,2], test_t[:,2].data.numpy())
performance[phase]['error'].append(error)
performance[phase]['loss'].append(loss.item())
performance[phase]['err_tdoa'].append(error1)
performance[phase]['err_itdoa'].append(error2)
performance[phase]['err_tdoe'].append(error3)
print('\nTest error',  error)

print(performance)

print('Saving to files:')
import pickle
output_dir = './results/'
filename = 'performance.pkl'
filehandler = open(output_dir + filename, 'wb')
pickle.dump(performance, filehandler, pickle.HIGHEST_PROTOCOL)


print('done? yes!')
