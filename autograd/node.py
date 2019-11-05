from grad_defs import grad_definitions

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
			self.grad_fns = grad_definitions[fn.__class__.__name__]
			
		except KeyError:
			fn_name = getattr(fn, '__name__', fn)
			raise NotImplementedError("Grad of {} wrt argnums {} not defined"
									  .format(fn_name))
