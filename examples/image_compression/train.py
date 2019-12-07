from autoencoder import Autoencoder
from flameflower.autograd import Tensor
from download_mnist import load_mnist

import numpy as np
import matplotlib.pyplot as plt

import flameflower as ff
import flameflower.nn as nn
import flameflower.nn.loss as loss
import flameflower.optim as optim
import flameflower.autograd.tensor_library as tl

X, y, X_test, y_test = load_mnist()

NUM_EXAMPLES = len(y)
INPUT_SIZE   = 784
HIDDEN_DIM   = 32
BATCH_SIZE   = 64
NUM_EPOCHS   = 10
NUM_ITERS    = NUM_EXAMPLES // BATCH_SIZE

X = X.reshape(len(X), INPUT_SIZE)
X = np.divide(X, 255)
X = Tensor(X)


model = Autoencoder(INPUT_SIZE, INPUT_SIZE, hidden_dim=HIDDEN_DIM)
loss_fn = loss.binary_cross_entropy
optimizer = optim.SGD(model.params(), lr=1e-3)

for epoch in range(NUM_EPOCHS):
	for i in range(NUM_ITERS):
		if i % 50 == 0 and i != 0:
			print(f"iter: {i}, loss: {loss_val.data}")
		sidx = i * BATCH_SIZE
		eidx = (i + 1) * BATCH_SIZE
		batch = X[sidx:eidx]
		reconstructed = model(batch)

		y_hat = tl.reshape(batch, (1, INPUT_SIZE * BATCH_SIZE))
		y = tl.reshape(reconstructed, (1, INPUT_SIZE * BATCH_SIZE))
		loss_val = loss_fn(y_hat, y)
		loss_val.backward()
		optimizer.step()
