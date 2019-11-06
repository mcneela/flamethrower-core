from __future__ import absolute_import
from node import GradNode
from utils import sum_with_none, topological_sort

import numpy as np

class Variable(object):
	type_mappings = {}
	types = set()

	__slots__ = ['_data', '_node']

	def __init__(self, data, node=None, track=True):
		self._data = data
		self._node  = node
		if not node and track:
			self._node = GradNode.new_root()

	def data(self):
		return self._data

	def node(self):
		return self._node

	def is_tracked(self):
		return self._is_tracked

	def __bool__(self):
		return bool(self._data)

	__nonzero__ = __bool__

	def backward(self):
		end_node = self.node()
		x = np.ones_like(self.data())
		outgrads = {end_node : x}
		for node in topological_sort(end_node):
			g = outgrads.pop(node)
			fun, value, args, kwargs, argnums = node.package
			for argnum, parent in zip(argnums, node.parents):
				grad = node.grad_fns[argnum]
				parent_grad = grad(value, g, *args, **kwargs)
				outgrads[parent] = sum_with_none(outgrads.get(parent), parent_grad)
				parent.grad = outgrads[parent]
		return g 

	@classmethod
	def register(cls, value_type):
		Variable.types.add(cls)
		Variable.type_mappings[value_type] = cls
