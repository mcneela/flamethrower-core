import numpy as np

import flameflower.nn as nn
import flameflower.optim as optim
import flameflower.autograd.tensor_library as tl

from flameflower.autograd import Tensor


X = np.linspace(-10, 10, num=100)
y_hat = np.sin(X)

X_train = Tensor(X)

class Network(nn.Module):
	def __init__(self, num_in, num_out):
		super(Network, self).__init__()
		self.num_in = num_in
		self.num_out = num_out
		self.lin1 = nn.Linear(num_in, 100)
		self.lin2 = nn.Linear(100, num_out)
		self.add_module('linear1', self.lin1)
		self.add_module('linear2', self.lin2)

	def _forward(self, X):
		out = self.lin1(X)
		out = self.lin2(out)
		return out

model = Network(1, 1)
optimizer = optim.SGD(model.params())
num_iters = 1000
loss_fn = lambda y, y_hat: (y_hat - y)

for i in range(num_iters):
	batch = Tensor([X[i]])
	y = model(batch)
	loss = loss_fn(y, y_hat)
	loss.data().resize(100, 1)
	loss.backward()
	optimizer.step()
