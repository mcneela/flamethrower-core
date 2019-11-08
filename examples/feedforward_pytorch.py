import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

np.random.seed(0)

X = np.linspace(-10, 10, num=100)
y_hat = np.sin(X)

X_train = torch.Tensor(X)

class Network(nn.Module):
	def __init__(self, num_in, num_out):
		super(Network, self).__init__()
		self.num_in = num_in
		self.num_out = num_out
		self.lin1 = nn.Linear(num_in, 5)
		self.lin2 = nn.Linear(5, num_out)

	def forward(self, X):
		out = self.lin1(X)
		out = self.lin2(out)
		return out

model = Network(1, 1)
optimizer = optim.SGD(model.parameters(), lr=1e-4)
num_iters = 1000
loss_fn = lambda y, y_hat: (y_hat - y) ** 2

for i in range(num_iters):
	print(i)
	break
	optimizer.zero_grad()
	batch = torch.Tensor([X[i % len(X)]])
	y = model(batch)
	loss = loss_fn(y, torch.Tensor([y_hat[i % len(y_hat)]]))
	# loss.node().name = '(y_hat - y)^2'
	# loss.data().resize(100, 1)
	loss.backward()
	optimizer.step()
	# print("W1 grad is: {} ".format(model.lin1.W.node().grad))

out = []
for j in range(len(y_hat)):
	z = model(torch.Tensor([X[j]])).data[0]
	out.append(z)
out = np.array(out)
plt.plot(X, out)
plt.show()

