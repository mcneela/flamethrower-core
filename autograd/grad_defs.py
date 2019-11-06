import numpy as np
from utils import name
from autograd import grad
from itertools import count

grad_definitions = {}
grad_defs_by_parity = {}

def define_grad(fn, *grads, **kwargs):
	for argnum, partial in enumerate(grads):
		if name(fn) not in grad_definitions:
			grad_definitions[name(fn)] = {}
		grad_definitions[name(fn)][argnum] = partial

def parity(fn):
	return fn.__code__.co_argcount

def record_grad_defs_by_parity(grad_definitions):
	for prim, grad_fn in grad_definitions:
		p = parity(grad_fn)
		if p in grad_defs_by_parity:
			grad_defs_by_parity[p].append(grad_fn)
		else:
			grad_defs_by_parity[p] = [grad_fn]
	return grad_defs_by_parity

# Multivariate gradients
define_grad(np.add,			lambda ans, g, x, y : g, lambda ans, g, x, y :  g)
define_grad(np.multiply,	lambda ans, g, x, y : g * y, lambda ans, g, x, y :  g * x)
define_grad(np.subtract,	lambda ans, g, x, y : g, lambda ans, g, x, y: -g)
define_grad(np.divide,		lambda ans, g, x, y : g / y, lambda ans, g, x, y:  g * -x / y**2)

# Single variable gradients
define_grad(np.exp,			lambda ans, g, x: g * ans)
define_grad(np.exp2, 		lambda ans, g, x: g * ans * np.log(2))
define_grad(np.expm1,		lambda ans, g, x: g * (ans + 1))
define_grad(np.log2,		lambda ans, g, x: g / (x * np.log(2)))
define_grad(np.log10, 		lambda ans, g, x: g / (x * np.log(10)))
define_grad(np.log1p, 		lambda ans, g, x: g / (x + 1))
define_grad(np.sin, 		lambda ans, g, x: g * np.cos(x))
define_grad(np.cos, 		lambda ans, g, x: g * -np.sin(x))
define_grad(np.tan, 		lambda ans, g, x: g / (np.cos(x) ** 2))
define_grad(np.arcsin, 		lambda ans, g, x: g / np.sqrt(1 - x ** 2))
define_grad(np.arccos, 		lambda ans, g, x: -g / np.sqrt(1 - x ** 2))
define_grad(np.arctan, 		lambda ans, g, x: g / np.sqrt(1 + x ** 2))
define_grad(np.sinh, 		lambda ans, g, x: g * np.cosh(x))
define_grad(np.cosh,		lambda ans, g, x: g * np.sinh(x))
define_grad(np.tanh,		lambda ans, g, x: g / (np.cosh(x) ** 2))
define_grad(np.arcsinh, 	lambda ans, g, x: g / np.sqrt(x ** 2 + 1))
define_grad(np.arccosh, 	lambda ans, g, x: g / np.sqrt(x ** 2 - 1))
define_grad(np.arctanh, 	lambda ans, g, x: g / np.sqrt(1 - x ** 2))
define_grad(np.rad2deg, 	lambda ans, g, x: g * 180 / np.pi)
define_grad(np.degrees, 	lambda ans, g, x: g * 180 / np.pi)
define_grad(np.deg2rad, 	lambda ans, g, x: g * np.pi / 180)
define_grad(np.radians, 	lambda ans, g, x: g * np.pi / 180)
define_grad(np.square, 		lambda ans, g, x: g * 2 * x)
define_grad(np.sqrt, 		lambda ans, g, x: g * 0.5 * x ** -0.5)
define_grad(np.sinc, 		lambda ans, g, x: g * ((np.cos(np.pi * x) / x) - (np.sin(np.pi * x))/(np.pi * (x**2))))
define_grad(np.reshape,		lambda ans, g, x, shape, order=None : np.reshape(g, np.shape(x), order=order))

def grad_transpose(ans, x, g, axes=None):
	if axes is not None:
		axes = np.argsort(axes)
	return np.transpose(g, axes)
define_grad(np.transpose, grad_transpose)



