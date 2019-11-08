"""
This file contains helpful utility functions
for neural network parameter initialization schemes.

(C) 2019 MLephant, FlameFlower
"""

import autograd.numpy as np
import autograd.numpy.random as npr

def normalized_initialization(in_size, out_size, size=None):
	"""
	See Glorot and Bengio (2010)
	Pg. 253, Eq. (16)
	"""
	high = np.sqrt(6 / (in_size + out_size))
	low = -high
	return npr.uniform(low, high, size=size)
