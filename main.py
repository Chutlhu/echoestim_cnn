import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from torch.autograd import Variable


filename = 'vast_picnic_task10_noise_vars.mat'

# HYPER-PARAMS
n_test = 200 # dimension of test set
n_epochs = 2000 # or whatever
batch_size = 128 # or whatever - NOT USED
H = 50 # hidden layer dimension

## IMPORT DATA 
data = scipy.io.loadmat('data/' + filename)

tdoa = data['tdoa']
N,_ = tdoa.shape

ILD  = data['ILD']
iIPD = data['iIPD']
rIPD = data['rIPD']

iIPD = np.hstack([np.zeros([N,1]),data['iIPD']])
rIPD = np.hstack([np.zeros([N,1]),data['rIPD']])

var = tdoa   # N samples
obs = np.stack([ILD, rIPD,iIPD], axis=2) # Fx3xN matrices
n_obs,n_feat,n_dims = obs.shape
n_train = n_obs - n_test

# Training set
random_indeces = np.random.permutation(n_obs)
train_var = var[random_indeces[0:n_train]]
train_obs = obs[random_indeces[0:n_train],:]
# Test set
test_var = var[random_indeces[n_train:n_obs]]
test_obs = obs[random_indeces[n_train:n_obs],:]


t = Variable(torch.from_numpy(train_var)).float()
y = Variable(torch.from_numpy(train_obs)).float()
print(t.shape)
print(y.shape)

N, F, D = y.data.numpy().shape
_, L = t.data.numpy().shape

xmin, xmax = np.min(tdoa)-0.0005, np.max(tdoa)+0.0005
ymin, ymax = xmin, xmax

## NETWORK
class ConvNet_Chakrabarty(torch.nn.Module):
    def __init__(self, n_filter, kernel_size, n_feature, n_hidden, n_output, dropout, classes):
        super(ConvNet_Chakrabarty, self).__init__()

        self.n_feature = n_feature
        self.dp = dropout

        self.conv1 = nn.Sequential(
            # conv2d: in_channels, out_channels, kernel_size
            nn.Conv2d(1, n_filter, kernel_size=kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=kernel_size))
        self.conv2 = nn.Sequential(
            nn.Conv2d(n_filter, n_filter, kernel_size=kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=kernel_size))
        self.conv3 = nn.Sequential(
            nn.Conv2d(n_filter, n_filter, kernel_size=kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=kernel_size))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(n_feature, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, self.n_feature)
        x = F.relu(self.drop_out(self.fc1(x)),p=self.dp)
        x = self.fc2(x)
        return x

class ConvNet_Girin(torch.nn.Module):
    def __init__(self):
        super(ConvNet_Girin, self).__init__()

        self.conv1 = torch.nn.Sequential(
            # conv2d: in_channels, out_channels, kernel_size
            torch.nn.Conv2d(3, 24, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2))
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(24, 48, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2))
        self.drop_out = torch.nn.Dropout()
        self.fc1 = torch.nn.Linear(12384, 360)
        self.fc2 = torch.nn.Linear(360, 240)
        self.fc3 = torch.nn.Linear(240, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, 12384)
        x = F.relu(self.drop_out(self.fc1(x)),p=0.3)
        x = F.relu(self.drop_out(self.fc2(x)),p=0.3)
        x = self.fc3(x)
        return x

net = ConvNet_Girin()    # define the network
print(net)  # net architecture

optimizer = torch.optim.Adam(net.parameters())
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

plt.ion()

## TRAINING
for it in range(n_epochs):
    prediction = net(y)     # input x and predict based on x

    loss = loss_func(prediction, t)     # must be (1. nn output, 2. target)

    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()         # backpropagation, compute gradients
    optimizer.step()        # apply gradients

    if it % 20 == 0 or it == n_epochs-1:
        # plot and show learning process
        plt.cla()
        axes = plt.gca()
        axes.set_xlim([xmin,xmax])
        axes.set_ylim([ymin,ymax])
        plt.scatter(t.data.numpy(), prediction.data.numpy())
        plt.plot(t.data.numpy(), t.data.numpy(), color='red')
        plt.text(0.0011, 0.001, r'%d/%d'%(it+1,n_epochs))
        plt.pause(0.1)

plt.ioff()
plt.close('all')

## PERFORMANCE METRICS
def nrmse(x, x_ref):
    return np.sqrt(np.sum((x - x_ref)**2, axis = 0))\
           /np.sqrt(np.sum((x_ref - np.mean(x_ref))**2, axis = 0))

train_error = nrmse(prediction.data.numpy(), t.data.numpy())
print('\nTrain error', train_error)

t = Variable(torch.from_numpy(test_var)).float()
y = Variable(torch.from_numpy(test_obs)).float()
prediction = net(y)     # input x and predict based on x
test_error = nrmse(prediction.data.numpy(), t.data.numpy())
print('\nTest error',  test_error)

plt.cla()
axes = plt.gca()
axes.set_xlim([xmin,xmax])
axes.set_ylim([ymin,ymax])
plt.scatter(t.data.numpy(), prediction.data.numpy())
plt.plot(t.data.numpy(), t.data.numpy(), color='red')
plt.show()

print('done? yes!')