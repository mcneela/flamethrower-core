import flamethrower.autograd.tensor_library as tl
from .module import Module
from .utils import get_logger

import logging

logger = get_logger()

class Dropout(Module):
	def __init__(self, p=0.5, on=True):
		super(Dropout, self).__init__()
		if not 0 <= p <= 1:
			raise ValueError("Dropout probability must be between zero and one.")
		self.p = p
		# Preserve the old `on` constructor argument, but store mode in the same
		# is_training flag used by every other stateful Module.
		self.is_training = bool(on)

	@property
	def on(self):
		"""Compatibility alias for the module's training state."""
		return self.is_training

	@on.setter
	def on(self, value):
		self.is_training = bool(value)

	def test_mode(self):
		return self.eval()

	def train_mode(self):
		return self.train()

	def forward(self, X):
		logger.info(f"Using dropout on data: {X} with probability: {self.p}")
		if not self.is_training or self.p == 0:
			return X
		if self.p == 1:
			return X * 0

		# p is the probability of dropping a unit. Dividing retained units by
		# their keep probability preserves the expected activation at inference.
		keep_probability = 1 - self.p
		mask = tl.random.uniform(0, 1, size=X.shape) >= self.p
		return X * mask / keep_probability

class L2Regularizer(Module):
	def __init__(self, weights, scale=1):
		super(L2Regularizer, self).__init__()
		try:
			iter(weights)
		except TypeError:
			weights = [weights]
		self.weights = weights
		self.scale = scale

	def forward(self):
		logger.info(f"Using L2 Regularization")
		term = 0
		for w in self.weights:
			term += tl.sum(tl.square(w))
		return self.scale * term

class L1Regularizer(Module):
	def __init__(self, weights, scale=1):
		super(L1Regularizer, self).__init__()
		try:
			iter(weights)
		except TypeError:
			weights = [weights]
		self.weights = weights
		self.scale = scale

	def forward(self):
		logger.info(f"Using L1 Regularization")
		term = 0
		for w in self.weights:
			term += tl.sum(tl.abs(w))
		return self.scale * term

class ElasticNetRegularizer(Module):
	def __init__(self, weights, lambda1=0.5, lambda2=0.5):
		super(ElasticNetRegularizer, self).__init__()
		try:
			iter(weights)
		except TypeError:
			weights = [weights]
		self.weights = weights
		self.lambda1 = lambda1
		self.lambda2 = lambda2

	def forward(self):
		logger.info("Using elastic net regularization.")
		term = 0
		for w in self.weights:
			term += self.lambda1 * tl.sum(tl.abs(w)) \
			      + self.lambda2 * tl.sum(tl.square(w))
		return term

def label_smoother(labels, eps=0.05):
	K = len(labels)
	labels[labels == 0] = eps / (K - 1)
	labels[labels == 1] = 1 - eps
	return labels


		
