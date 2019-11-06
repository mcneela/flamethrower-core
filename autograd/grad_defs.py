import numpy as np
from autograd import grad
from itertools import count

grad_definitions = {}
grad_defs_by_parity = {}

def name(fn):
	return fn.__name__()

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
define_grad(np.add,			lambda ans, x, y :  1, lambda ans, x, y :  1)
define_grad(np.multiply,	lambda ans, x, y : y, lambda ans, x, y :  x)
define_grad(np.subtract,	lambda ans, x, y : 1, lambda ans, x, y: -1)
define_grad(np.divide,		lambda ans, x, y : 1 / y, lambda ans, x, y:  -x / y**2)

# Single variable gradients
define_grad(np.exp,			lambda ans, x: ans)
define_grad(np.exp2, 		lambda ans, x: ans * np.log(2))
define_grad(np.expm1,		lambda ans, x: ans + 1)
define_grad(np.log2,		lambda ans, x: 1 / (x * np.log(2)))
define_grad(np.log10, 		lambda ans, x: 1 / (x * np.log(10)))
define_grad(np.log1p, 		lambda ans, x: 1 / (x + 1))
define_grad(np.sin, 		lambda ans, x: np.cos(x))
define_grad(np.cos, 		lambda ans, x: -np.sin(x))
define_grad(np.tan, 		lambda ans, x: 1 / (np.cos(x) ** 2))
define_grad(np.arcsin, 		lambda ans, x: 1 / np.sqrt(1 - x ** 2))
define_grad(np.arccos, 		lambda ans, x: -1 / np.sqrt(1 - x ** 2))
define_grad(np.arctan, 		lambda ans, x: 1 / np.sqrt(1 + x ** 2))
define_grad(np.sinh, 		lambda ans, x: np.cosh(x))
define_grad(np.cosh,		lambda ans, x: np.sinh(x))
define_grad(np.tanh,		lambda ans, x: 1 / (np.cosh(x) ** 2))
define_grad(np.arcsinh, 	lambda ans, x: 1 / np.sqrt(x ** 2 + 1))
define_grad(np.arccosh, 	lambda ans, x: 1 / np.sqrt(x ** 2 - 1))
define_grad(np.arctanh, 	lambda ans, x: 1 / np.sqrt(1 - x ** 2))
define_grad(np.rad2deg, 	lambda ans, x: 180 / np.pi)
define_grad(np.degrees, 	lambda ans, x: 180 / np.pi)
define_grad(np.deg2rad, 	lambda ans, x: np.pi / 180)
define_grad(np.radians, 	lambda ans, x: np.pi / 180)
define_grad(np.square, 		lambda ans, x: 2 * x)
define_grad(np.sqrt, 		lambda ans, x: 0.5 * x ** -0.5)
define_grad(np.sinc, 		lambda ans, x : (np.cos(np.pi * x) / x) - (np.sin(np.pi * x))/(np.pi * (x**2)))




