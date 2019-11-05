from collections import OrderedDict
import logging
import pickle

class Module(object):
	__slots__ = ['logger', 'is_training', '_buffers', '_params' \
				 '_children', '_state', '_modules']
	def __init__(self, name=None):
		if name:
			self.__name__ = name
		self.logger = logging.getLogger('newModel')
		self.is_training = True
		# Buffers are persistent state objects
		# but not module parameters
		self._buffers = OrderedDict()
		self._params = OrderedDict()
		self._children = OrderedDict()
		self._state = OrderedDict()
		self._modules = OrderedDict()

	def new_buffer(self, name, obj):
		self._buffers[name] = obj

	def new_param(self, name, param):
		self._params[name] = param

	def add_module(self, name, module):
		self._modules[name] = module

	def _forward(self, *args, **kwargs):
		raise NotImplementedError

	def __call__(self, *args, **kwargs):
		return self._forward(*args, **kwargs)

	def params(self):
		for name, param in self._params:
			yield param

	def children(self):
		for name, child in self._children:
			yield child

	def train(self, train_mode=True):
		self.is_training = train_mode 
		for child in self.children():
			child.train(train_mode)
		return self

	def infer(self):
		return self.train(train_mode=False)

	def zero_grad(self):
		for param in self.params():
			if param.grad:
				param.grad.detach_()
				param.grad.zero_()

	def 