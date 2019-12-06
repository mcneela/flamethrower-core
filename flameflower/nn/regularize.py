import flameflower.autograd.tensor_library as tl
from .module import Module

class Dropout(Module):
	def __init__(self, p=0.5, on=True):
		super(Dropout, self).__init__()
		self.p = p
		self.on = on

	def test_mode(self):
		self.on = False

	def train_mode(self):
		self.on = True

	def forward(self, X):
		if not self.on:
			return X
		mask = tl.random.uniform(0, 1, size=X.shape)
		mask = mask < self.p
		return X * mask

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
		term = 0
		for w in self.weights:
			term += 0.5 * tl.sum(tl.square(w))
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
		term = 0
		for w in self.weights:
			term += tl.sum(tl.abs(w))
		return self.scale * term
		