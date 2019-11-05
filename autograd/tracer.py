from grad_defs import grad_definitions
import tensor_library as tl
from tensor import Tensor
import sys

def trace(start_node, fn, x):
	start_box = Tensor(x, start_node)
	end_box = fn(start_box)
	return end_box._data, end_box._node

class Node(object):
	__slots__ = []
	def __init__(self, val, fn, args, kwargs, argnums, parents):
		assert False

	def initialize_root(self, *args, **kwargs):
		assert False

	@classmethod
	def new_root(cls, *args, **kwargs):
		root = cls.__new__(cls)
		root.initialize_root(*args, **kwargs)
		return root

class GradNode(Node):
	__slots__ = ['parents', 'grad', 'grad_fns', 'recipe']
	def __init__(self, val, fn, args, kwargs, argnums, parents):
		self.parents = parents
		self.recipe = fn, val, args, kwargs, argnums
		try:
			self.grad_fns = grad_definitions[fn.fn.__name__]

		except KeyError:
			fn_name = getattr(fn, '__name__', fn)
			raise NotImplementedError("Grad of {} wrt argnums {} not defined"
									  .format(fn_name))

	def initialize_root(self):
		self.parents = []
		self.recipe = (lambda x: x, None, (), {}, [])
		self.grad_fns = {0: lambda g, ans, x, y: (), 1: lambda g, ans, x, y: ()}

