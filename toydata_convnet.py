### proza ###
import torch
import torch.nn.functional as F


dtype = torch.float
device = torch.device("cpu")

N, C, W, H, D_out = 64, 1, 256, 3, 10

num_epochs = 5
num_classes = 10
batch_size = N
learning_rate = 0.001

inputs = torch.randn(N, C, W, H, device=device, dtype=dtype)
target = torch.randn(N, D_out,  device=device, dtype=dtype)

class ConvNet(torch.nn.Module):
	def __init__(self):
		super(ConvNet, self).__init__()
		self.conv1 = torch.nn.Conv2d(1,  20, kernel_size = 3, stride = 1, padding = 1)
		self.conv2 = torch.nn.Conv2d(20, 50, kernel_size = 3, stride = 1, padding = 1)
		self.fc1 = torch.nn.Linear(1250, 500)
		self.fc2 = torch.nn.Linear(500, 10)

	def forward(self, x):
		x = F.relu(self.conv1(x))
		print('F.relu(self.conv1(x))', x.shape)
		x = F.max_pool2d(x, 2, 2)
		print('F.max_pool2d(x, 2, 2)', x.shape)
		x = F.relu(self.conv2(x))
		print('F.relu(self.conv2(x))', x.shape)
		x = F.max_pool2d(x, 2, 2)
		print('F.max_pool2d(x, 2, 2)', x.shape)
		x = x.view(-1, 1250)
		print('x.view(-1, 1250)', x.shape)
		x = F.relu(self.fc1(x))        
		print('F.relu(self.fc1(x))', x.shape)
		x = self.fc2(x)
		print('self.fc2(x)', x.shape)
		x = F.log_softmax(x, dim=1)
		print('F.log_softmax(x, dim=1)', x.shape)
		return x

model = ConvNet()

# Loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


model.train()
optimizer.zero_grad()

output = model(inputs)
print(output.shape)
print(target.shape)
loss = criterion(output, target)
loss.backward()
optimizer.step()

