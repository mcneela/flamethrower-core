import tensor_library as tl
from utils import name
from grad_defs import grad_definitions

class Node(object):
	__slots__ = []
	def __init__(self, val, fn, args, kwargs, argnums, parents):
		raise NotImplementedError

	def init_root(self, *args, **kwargs):
		raise NotImplementedError

	@classmethod
	def new_root(cls, *args, **kwargs):
		root = cls.__new__(cls)
		root.init_root(*args, **kwargs)
		return root

class GradNode(Node):
	__slots__ = ['parents', 'grad', 'grad_fns', 'recipe']
	def __init__(self, val, fn, args, kwargs, argnums, parents):
		self.parents = parents
		self.recipe = fn, val, args, kwargs, argnums
		try:
			self.grad_fns = grad_definitions[name(fn)]
		except KeyError:
			fn_name = getattr(fn, '__name__', fn)
			raise NotImplementedError("Grad of {} wrt argnums {} not defined"
									  .format(fn_name))

	def init_root(self):
		self.parents = []
		self.recipe = (lambda x: x, None, (), {}, [])
		self.grad_fns = {0: lambda g, ans, x, y: (), 1: lambda g, ans, x, y: ()}

