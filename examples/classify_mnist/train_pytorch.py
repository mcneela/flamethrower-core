from __future__ import division

import numpy as np

import flameflower.nn as nn
import flameflower.nn.loss as loss
import flameflower.optim as optim
import flameflower.autograd.tensor_library as tl

from flameflower.autograd import Tensor
import matplotlib.pyplot as plt

from download_mnist import load_mnist
from feedforward import FeedforwardNetwork

np.random.seed(0)

X, y, X_test, y_test = load_mnist()

NUM_EXAMPLES = len(y)
DATA_SIZE = 784
BATCH_SIZE = 64
NUM_CLASSES = 10
NUM_EPOCHS = 5
NUM_ITERS = NUM_EXAMPLES // BATCH_SIZE

X = X.reshape(len(X), DATA_SIZE)
X = np.divide(X, 255)
model = FeedforwardNetwork(DATA_SIZE, NUM_CLASSES)
optimizer = optim.SGD(model.params(), lr=1e-4)
num_iters = 1000
loss_fn = loss.cross_entropy

for epoch in range(NUM_EPOCHS):
	np.random.shuffle(X)
	print(f"Current Epoch: {epoch}")
	for i in range(NUM_ITERS):
		if i % 50 == 0:
			print(f"Current iteration: {i}")
		sidx = i * BATCH_SIZE
		eidx = (i + 1) * BATCH_SIZE
		batch = Tensor(X[sidx:eidx])
		preds = tl.argmax(model(batch), axis=1)
		y_hat = y[sidx:eidx]
		loss = loss_fn(y_hat, preds)
		loss.backward()
		optimizer.step()
