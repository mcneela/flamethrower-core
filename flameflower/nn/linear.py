from .module import Module
from flameflower.autograd import Tensor

import flameflower.autograd.tensor_library as tl
import flameflower.autograd.tensor_library.random as tlr
import flameflower.nn.initialize as init

tlr.seed(0)
class Linear(Module):
	def __init__(self, in_size, out_size, use_bias=True):
		super(Linear, self).__init__()
		self.in_size = in_size
		self.out_size = out_size
		self.use_bias = use_bias
		self._init_params()

	def _init_params(self, init_fn=None):
		if not init_fn:
			init_fn = init.xavier_normal
		self.W = Tensor(init_fn(self.in_size, self.out_size))
		self.b = Tensor(tl.zeros((1, self.W.shape[1])))
		self.new_param('W', self.W)
		if self.use_bias:
			self.b = Tensor(tl.ones((1, self.W.shape[1])))
			self.new_param('b', self.b)

	def forward(self, X):
		if self.use_bias:
			return X @ self.W + self.b
		else:
			return X @ self.W

