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
