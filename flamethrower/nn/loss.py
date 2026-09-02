"""
A collection of loss
functions used in training
deep neural networks.
"""
from __future__ import division
import numpy as np
import flamethrower.autograd.tensor_library as tl

def cross_entropy(y, y_hat, regularizer=None):
	"""
	Cross-entropy loss, used for
	classification with n class.
	"""
	if regularizer is None:
		regularizer = lambda: 0
	total = 0
	z = log_softmax(y_hat, axis=1)
	for i, label in enumerate(y):
		total += -z[i][label]
	return total / len(y) + regularizer()

def log_softmax(x, axis=None):
	b = tl.max(x, axis=axis, keepdims=True)
	return (x - b) - tl.log(tl.sum(tl.exp(x - b), axis=axis, keepdims=True))
	
def binary_cross_entropy(y, y_hat, eps=1e-5, regularizer=None):
	"""
	Binary cross-entropy loss, used
	for classification with 2 classes.
	"""
	if regularizer is None:
		regularizer = lambda: 0
	# Clipping the prediction directly into [eps, 1-eps] keeps log()'s
	# argument in a safe range without disturbing it anywhere else. Adding
	# eps inside every log call instead let a prediction of exactly 1.0 push
	# log(y_hat+eps) slightly above log(1), producing a small NEGATIVE loss
	# for a perfect prediction - impossible for a true loss - and it added
	# the same eps-sized error to every gradient, not just ones near the
	# boundary that actually need clipping.
	clipped = tl.where(y_hat.data < eps, eps, tl.where(y_hat.data > 1 - eps, 1 - eps, y_hat))
	return tl.mean(-y * tl.log(clipped) - (1 - y) * tl.log(1 - clipped)) + regularizer()

def mean_squared_error(y, y_hat, regularizer=None):
	"""
	MSE loss, often used for
	regression tasks.
	"""
	if regularizer is None:
		regularizer = lambda: 0
	n = len(y)
	l = (1 / n) * tl.sum((y_hat - y) ** 2)
	return l + regularizer()

def l2(y, y_hat, regularizer=None, eps=1e-12):
	"""
	Loss function using the L2 (Euclidean) norm of the residual vector.
	Equivalent to minimization with MSE loss, since sqrt is monotonic on
	nonnegative inputs and MSE minimizes the same squared norm.

	`eps` keeps the gradient finite when the residual norm is exactly zero -
	d/dx sqrt(x) is undefined at x=0, which sum-of-squares reaches whenever
	every prediction in the batch exactly matches its target.
	"""
	if regularizer is None:
		regularizer = lambda: 0
	return tl.sqrt(tl.sum((y_hat - y) ** 2) + eps) + regularizer()

def l1(y, y_hat, regularizer=None):
	"""
	Loss function using the L1 (Manhattan) norm of the residual vector.
	"""
	if regularizer is None:
		regularizer = lambda: 0
	return tl.sum(tl.abs(y_hat - y)) + regularizer()

def kl_divergence(p, q, regularizer=None):
	"""
	Returns a notion of "distance" between two probability distributions
	p and q. `p` is treated as a fixed reference distribution (a plain
	array, like the labels in the loss functions above); only `q` needs to
	be a Tensor to backpropagate through.
	"""
	if regularizer is None:
		regularizer = lambda: 0
	p = np.asarray(p)
	# p*log(q/p) = p*log(q) - p*log(p). Dividing by p directly, as q/p did,
	# produced inf*0 = NaN at any p entry that was exactly zero - even
	# though a zero-probability entry's true contribution to KL divergence
	# is defined to be zero (the same 0*log(0) = 0 convention used for
	# entropy). Splitting the ratio this way avoids ever dividing by p.
	safe_p = np.where(p == 0, 1, p)
	p_log_p = np.sum(np.where(p == 0, 0, p * np.log(safe_p)))
	return p_log_p - tl.sum(p * tl.log(q)) + regularizer()

def huber(y, y_hat, delta=1, regularizer=None):
	"""
	Huber loss: quadratic for small residuals, linear beyond `delta`.
	"""
	if regularizer is None:
		regularizer = lambda: 0
	residual = y_hat - y
	# `tl.abs(residual) < delta` is a Tensor, and Python's `if` can only
	# branch on a single value - it raised for any batch with more than one
	# element. tl.where makes the choice per element instead; the condition
	# itself is kept a raw array (residual.data) rather than a traced Tensor
	# comparison, same as in activations.py.
	quadratic = 0.5 * residual ** 2
	# The standard Huber formula is delta*(|r| - 0.5*delta), i.e. delta*|r| -
	# 0.5*delta**2. Subtracting delta/2 instead only happens to be correct at
	# the default delta=1 (where 0.5*delta**2 == delta/2); any other delta
	# gave the wrong constant.
	linear = delta * tl.abs(residual) - 0.5 * delta ** 2
	return tl.where(abs(residual.data) < delta, quadratic, linear) + regularizer()

def huber_binary_loss(y, y_hat, delta=1, regularizer=None):
	"""
	Modified Huber loss for binary classification with labels in {-1, +1}
	(0 is treated as the negative class and remapped to -1).
	"""
	if regularizer is None:
		regularizer = lambda: 0
	# Only y (the label) is remapped from {0, 1} to {-1, +1}. y_hat is a
	# continuous prediction score, and 0.0 is a meaningful value there in
	# its own right - e.g. right after initialization - not a stand-in for
	# a negative label, so it must not be remapped the same way.
	y = np.where(np.asarray(y) == 0, -1, y)
	margin = y_hat * y
	# tl.max reduces an array to its largest element (numpy's second
	# positional arg to max() is `axis`, not a value to compare against);
	# the elementwise max against 0 needs tl.maximum instead.
	quadratic = tl.maximum(0, 1 - margin) ** 2
	linear = -4 * margin
	# regularizer() was computed above but never actually added to either
	# branch in the original code, unlike every other loss in this file.
	return tl.where(margin.data >= -1, quadratic, linear) + regularizer()

