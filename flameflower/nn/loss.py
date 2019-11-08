"""
A collection of loss
functions used in training
deep neural networks.

(C) 2019 - MLephant, FlameFlower
"""
from __future__ import division

import flameflower.autograd.tensor_library as tl

def cross_entropy(y_hat, y):
	return -tl.sum(y * tl.log(y_hat))

def binary_cross_entropy(y_hat, y):
	return -y * tl.log(y_hat) - (1 - y) * tl.log(1 - y_hat)

def mean_squared_error(y_hat, y):
	n = len(y)
	l = (1 / n) * tl.sum((y_hat - y) ** 2)
	return l

def l2_loss(y_hat, y):
	return 0.5 * tl.sum((y_hat - y) ** 2)

def l1_loss(y_hat, y):
	return tl.abs(y_hat - y)

def kl_divergence(p, q):
	return -tl.sum(p * tl.log(q / p))

def huber_loss(y_hat, y, delta=1):
	if tl.abs(y_hat - y) < delta:
		return .5 * (y_hat - y) ** 2
	else:
		return delta * tl.abs(y_hat - y) - (delta / 2)
		