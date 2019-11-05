import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
from module import Module

class Linear(Module):
	def __init__(self, in_size, out_size, use_bias=True):
		super(Linear, self).__init__()
		self.in_size = in_size
		self.out_size = out_size
		self.use_bias = use_bias
		self._init_params()

	def _init_params(self, init_fn=None):
		if not init_fn:
			init_fn = npr.randn
		self.W = init_fn(self.in_size, self.out_size)
		self.b = np.zeros(self.out_size)
		self.new_param('W', self.W)
		if self.use_bias:
			self.b = np.ones(self.out_size)
			self.new_param('b', self.b)

	def _forward(self, x):
		return self.W @ x + self.b

