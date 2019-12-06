"""
A collection of loss
functions used in training
deep neural networks.

(C) 2019 - MLephant, FlameFlower
"""
from __future__ import division
import flameflower.autograd.tensor_library as tl

def cross_entropy(y_hat, y):
	"""
	Cross-entropy loss, used for
	classification with n classes.
	"""
	total = 0
	z = log_softmax(y, axis=1)
	for i, label in enumerate(y_hat):
		total += -z[i][label]
	return total / len(y_hat)

def log_softmax(x, axis=None):
	b = tl.max(x, axis=axis, keepdims=True)
	return (x - b) - tl.log(tl.sum(tl.exp(x - b), axis=axis, keepdims=True))
	
def binary_cross_entropy(y_hat, y):
	"""
	Binary cross-entropy loss, used
	for classification with 2 classes.
	"""
	return -y * tl.log(y_hat) - (1 - y) * tl.log(1 - y_hat)

def mean_squared_error(y_hat, y):
	"""
	MSE loss, often used for
	regression tasks.
	"""
	n = len(y)
	l = (1 / n) * tl.sum((y_hat - y) ** 2)
	return l

def l2(y_hat, y):
	"""
	Loss function using the L2 norm.
	Equivalent to minimization with
	MSE loss.
	"""
	return 0.5 * tl.sum((y_hat - y) ** 2)

def l1(y_hat, y):
	"""
	Loss function using the L1 norm.
	"""
	return tl.abs(y_hat - y)

def kl_divergence(p, q):
	"""
	Returns a notion of "distance"
	between two probability distributions
	p and q.
	"""
	return -tl.sum(p * tl.log(q / p))

def huber(y_hat, y, delta=1):
	if tl.abs(y_hat - y) < delta:
		return .5 * (y_hat - y) ** 2
	else:
		return delta * tl.abs(y_hat - y) - (delta / 2)
