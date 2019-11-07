from .utils import name
from .grad_defs import grad_definitions

class Node(object):
	"""
	Base class for a generic
	node type.
	"""
	__slots__ = []
	def __init__(self, val, fn, args, kwargs, argnums, parents):
		raise NotImplementedError

	def init_root(self, *args, **kwargs):
		raise NotImplementedError

	def is_root(self):
		return self._is_root

	@classmethod
	def new_root(cls, *args, **kwargs):
		root = cls.__new__(cls)
		root.init_root(*args, **kwargs)
		root._is_root = True
		return root

class GradNode(Node):
	"""
	A node that attaches to a `Variable`
	for which we want to take a gradient.
	"""
	__slots__ = ['parents', 'grad', 'grad_fns', 'package', '_is_root']
	def __init__(self, val, fn, args, kwargs, argnums, parents):
		self._is_root = False
		self.parents = parents
		self.package = fn, val, args, kwargs, argnums
		try:
			self.grad_fns = grad_definitions[fn.fn.__name__]
		except KeyError:
			fn_name = getattr(fn, '__name__', fn)
			raise NotImplementedError("Grad of {} wrt argnums {} not defined"
									  .format(fn_name, argnums))

	def init_root(self):
		self.parents = []
		self.package = (lambda x: x, None, (), {}, [])
		self.grad_fns = {0: lambda g, ans, x, y: (), 1: lambda g, ans, x, y: ()}

