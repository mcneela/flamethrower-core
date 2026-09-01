"""
A collection of loss
functions used in training
deep neural networks.
"""
from __future__ import division
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
	return tl.mean(-y * tl.log(y_hat + eps) - (1 - y) * tl.log(1 - y_hat + eps)) + regularizer()

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
	Returns a notion of "distance"
	between two probability distributions
	p and q.
	"""
	if regularizer is None:
		regularizer = lambda: 0
	return -tl.sum(p * tl.log(q / p)) + regularizer()

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
	linear = delta * tl.abs(residual) - (delta / 2)
	return tl.where(abs(residual.data) < delta, quadratic, linear) + regularizer()

def huber_binary_loss(y, y_hat, delta=1, regularizer=None):
	"""
	Modified Huber loss for binary classification with labels in {-1, +1}
	(0 is treated as the negative class and remapped to -1).
	"""
	if regularizer is None:
		regularizer = lambda: 0
	# As with huber() above, these were plain `if`s comparing Tensors, which
	# only worked for a single-element input; tl.where vectorizes them.
	y_hat = tl.where(y_hat.data == 0, -1, y_hat)
	y = tl.where(y.data == 0, -1, y)
	margin = y_hat * y
	# tl.max reduces an array to its largest element (numpy's second
	# positional arg to max() is `axis`, not a value to compare against);
	# the elementwise max against 0 needs tl.maximum instead.
	quadratic = tl.maximum(0, 1 - margin) ** 2
	linear = -4 * margin
	# regularizer() was computed above but never actually added to either
	# branch in the original code, unlike every other loss in this file.
	return tl.where(margin.data >= -1, quadratic, linear) + regularizer()

