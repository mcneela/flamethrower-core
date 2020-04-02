from __future__ import division

import numpy as np
import sys

import torch
import torch.nn as nn
import torch.optim as optim

import flamethrower.nn as nn2
import flamethrower.nn.loss as loss
import flamethrower.optim as optim2
import flamethrower.autograd.tensor_library as tl

import flamethrower.autograd as ag

from torch import Tensor
import matplotlib.pyplot as plt

from download_mnist import load_mnist
from feedforward_pt import FeedforwardNetwork
from feedforward import FeedforwardNetwork as FF2

# np.random.seed(0)
tl.random.seed(0)
np.random.seed(0)

X, y, X_test, y_test = load_mnist()

NUM_EXAMPLES = len(y)
DATA_SIZE = 784
BATCH_SIZE = 128
NUM_CLASSES = 10
NUM_EPOCHS = 10
NUM_ITERS = NUM_EXAMPLES // BATCH_SIZE

X = X.reshape(len(X), DATA_SIZE)
X = np.divide(X, 255)
X = X - np.mean(X, axis=0, keepdims=True)
z = np.std(X, axis=0, keepdims=True)
z[z == 0] = 1
X = X / z
model = FeedforwardNetwork(DATA_SIZE, NUM_CLASSES)
model2 = FF2(DATA_SIZE, NUM_CLASSES)
with torch.enable_grad():
	model.lin1.weight.data = torch.FloatTensor(model2.lin1.W.data)
	model.lin1.bias.data = torch.FloatTensor(model2.lin1.b.data).resize(200)
	model.lin2.weight.data = torch.FloatTensor(model2.lin2.W.data)
	model.lin2.bias.data = torch.FloatTensor(model2.lin2.b.data).resize(60)
	model.lin3.weight.data = torch.FloatTensor(model2.lin3.W.data)
	model.lin3.bias.data = torch.FloatTensor(model2.lin3.b.data).resize(10)
optimizer = optim.SGD(model.parameters(), lr=1e-3)
optimizer2 = optim2.SGD(model2.params(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()
loss_fn2 = loss.cross_entropy

X1 = Tensor(X)
X2 = ag.Tensor(X)
for epoch in range(NUM_EPOCHS):
	print(f"Current Epoch: {epoch}")
	# np.random.shuffle(X)
	for i in range(NUM_ITERS):
		if i % 50 == 0 and i != 0:
			print(f"Current iteration: {i}")
			print(f"Current loss: {loss.data}")
		sidx = i * BATCH_SIZE
		eidx = (i + 1) * BATCH_SIZE
		batch = X1[sidx:eidx]
		batch2 = X2[sidx:eidx]
		preds = model(batch)
		preds2 = model2(batch2)
		# sys.exit()
		y_hat = torch.LongTensor(y[sidx:eidx])
		y_hat2 = y[sidx:eidx]
		lossval = loss_fn(preds, y_hat)
		lossval2 = loss_fn2(y_hat2, preds2)
		lossval.backward()
		lossval2.backward()
		optimizer.step()
		optimizer2.step()
		break
	break
