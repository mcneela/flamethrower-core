import tensor as ten
import variable as var

def isvar(x):
	return isinstance(x, var.Variable)

def get_data(args, var_args):
	args = list(args)
	for i, arg in var_args:
		args[i] = arg._data
	return tuple(args)

def get_parents(var_args):
	if hasattr(var_args, '__iter__'):
		return tuple(x[1]._node for x in var_args)
	else:
		return tuple(x[1]._node)

def get_argnums(var_args):
	if hasattr(var_args, '__iter__'):
		return tuple(x[0] for x in var_args)
	else:
		return tuple(x[0])

class Primitive(object):
	def __init__(self, fn):
		self.fn = fn

	def __name__(self):
		try:
			return self.fn.__name__
		except AttributeError:
			return "FlameFlower primitive with undefined name"

	def __call__(self, *args, **kwargs):
		var_args, node_type = self.find_var_args(args)
		if var_args:
			argdata = get_data(args, var_args)
			parents = get_parents(var_args)
			argnums = get_argnums(var_args)
			ans     = self(*argdata, **kwargs)
			node    = node_type(ans, self, argdata, kwargs, argnums, parents)
			return  ten.Tensor(ans, node)
		else:
			return self.fn(*args, **kwargs)

	def find_var_args(self, args):
		var_args = []
		node_type = None
		for argnum, arg in enumerate(args):
			if isvar(arg):
				var_args.append((argnum, arg))
				node_type = type(arg._node)
		return var_args, node_type
