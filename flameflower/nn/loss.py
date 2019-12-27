"""
A collection of loss
functions used in training
deep neural networks.

(C) 2019 - MLephant, FlameFlower
"""
from __future__ import division
import flameflower.autograd.tensor_library as tl

def cross_entropy(y_hat, y, regularizer=None):
	"""
	Cross-entropy loss, used for
	classification with n classes.
	"""
	if regularizer is None:
		regularizer = lambda: 0
	total = 0
	z = log_softmax(y, axis=1)
	for i, label in enumerate(y_hat):
		total += -z[i][label]
	return total / len(y_hat) + regularizer()

def log_softmax(x, axis=None):
	b = tl.max(x, axis=axis, keepdims=True)
	return (x - b) - tl.log(tl.sum(tl.exp(x - b), axis=axis, keepdims=True))
	
def binary_cross_entropy(y_hat, y, eps=1e-5, regularizer=None):
	"""
	Binary cross-entropy loss, used
	for classification with 2 classes.
	"""
	if regularizer is None:
		regularizer = lambda: 0
	return tl.mean(-y_hat * tl.log(y + eps) - (1 - y_hat) * tl.log(1 - y + eps) + regularizer())

def mean_squared_error(y_hat, y, regularizer=None):
	"""
	MSE loss, often used for
	regression tasks.
	"""
	if regularizer is None:
		regularizer = lambda: 0
	n = len(y)
	l = (1 / n) * tl.sum((y_hat - y) ** 2)
	return l + regularizer()

def l2(y_hat, y, regularizer=None):
	"""
	Loss function using the L2 norm.
	Equivalent to minimization with
	MSE loss.
	"""
	if regularizer is None:
		regularizer = lambda: 0
	return 0.5 * tl.sum(tl.sqrt((y_hat - y) ** 2)) + regularizer()

def l1(y_hat, y, regularizer=None):
	"""
	Loss function using the L1 norm.
	"""
	if regularizer is None:
		regularizer = lambda: 0
	return tl.abs(y_hat - y) + regularizer()

def kl_divergence(p, q, regularizer=None):
	"""
	Returns a notion of "distance"
	between two probability distributions
	p and q.
	"""
	if regularizer is None:
		regularizer = lambda: 0
	return -tl.sum(p * tl.log(q / p)) + regularizer()

def huber(y_hat, y, delta=1, regularizer=None):
	if regularizer is None:
		regularizer = lambda: 0
	if tl.abs(y_hat - y) < delta:
		return .5 * (y_hat - y) ** 2 + regularizer()
	else:
		return delta * tl.abs(y_hat - y) - (delta / 2) + regularizer()
