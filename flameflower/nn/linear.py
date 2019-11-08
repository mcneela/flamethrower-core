from .module import Module
import flameflower.autograd.tensor_library as np
from flameflower.autograd import Tensor
import numpy.random as npr

npr.seed(0)
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
		self.W = Tensor(init_fn(self.in_size, self.out_size))
		self.b = Tensor(np.zeros((1,self.W.data().shape[1])))
		self.new_param('W', self.W)
		if self.use_bias:
			self.b = Tensor(np.ones((1,self.W.data().shape[1])))
			self.new_param('b', self.b)

	def forward(self, X):
		if self.use_bias:
			return X @ self.W + self.b
		else:
			return X @ self.W 

