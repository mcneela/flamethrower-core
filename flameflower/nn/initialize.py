"""
This file contains helpful utility functions
for neural network parameter initialization schemes.

(C) 2019 MLephant, FlameFlower
"""

import flameflower.autograd.tensor_library as tl
import flameflower.autograd.tensor_library.random as tlr

def normalized_initialization(in_size, out_size, size=None):
	"""
	See Glorot and Bengio (2010)
	Pg. 253, Eq. (16)
	"""
	high = tl.sqrt(6 / (in_size + out_size))
	low = -high
	return tlr.uniform(low, high, size=size)
