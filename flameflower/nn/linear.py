from .module import Module
from flameflower.autograd import Tensor

import flameflower.autograd.tensor_library as tl
import flameflower.autograd.tensor_library.random as tlr
import flameflower.nn.initialize as init

class Linear(Module):
	def __init__(self, in_size, out_size, use_bias=True):
		super(Linear, self).__init__()
		self.in_size = in_size
		self.out_size = out_size
		self.use_bias = use_bias
		self._init_params()

	def _init_params(self, init_fn=None):
		if not init_fn:
			init_fn = init.glorot_uniform
		self.W = Tensor(init_fn(self.out_size, self.in_size))
		self.b = Tensor(tl.zeros((1, self.W.shape[0])))
		self.new_param('W', self.W)
		if self.use_bias:
			self.b = Tensor(init_fn(1, self.W.shape[0]))
			self.new_param('b', self.b)

	def forward(self, X):
		out = X @ tl.transpose(self.W)
		if self.use_bias:
			out += self.b
		return out