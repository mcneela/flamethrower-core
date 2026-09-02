from __future__ import absolute_import

import flamethrower.autograd.node as anode
import flamethrower.autograd.utils as utils

import numpy as np

class Variable(object):
	type_mappings = {}
	types = set()

	__slots__ = ['_data', '_node', '_is_tracked']

	def __init__(self, data, node=None, track=True):
		self._data = data
		self._node = node
		if not node and track:
			self._node = anode.GradNode.new_root()
		self._is_tracked = track

	@property
	def data(self):
		return self._data

	@data.setter
	def data(self, data):
		self._data = data

	@property
	def node(self):
		return self._node

	@property
	def grad(self):
		if self._node and isinstance(self._node, anode.GradNode):
			return self._node._grad
		else:
			raise AttributeError("This Variable does not have a GradNode attached.")

	@property
	def is_tracked(self):
		return self._is_tracked

	@property
	def shape(self):
		return self._data.shape

	@property
	def dims(self):
		return self._data.ndim

	@property
	def numel(self):
		return self._data.size

	def __bool__(self):
		return bool(self._data)

	__nonzero__ = __bool__

	def backward(self):
		end_node = self.node
		x = np.ones_like(self.data)
		outgrads = {end_node : x}
		g = x
		for node in utils.topological_sort(end_node):
			# A NoGradNode (e.g. from a comparison like `x > 0`, or anything
			# built from one) never receives an incoming gradient - nothing is
			# allowed to write into it, see the parent-loop below - so it has
			# no entry here unless it's the end_node itself. Either way,
			# gradient does not flow through it, matching PyTorch's
			# detach()/stop_gradient.
			g = outgrads.pop(node, None)
			if isinstance(node, anode.NoGradNode) or g is None:
				continue
			fn, value, args, kwargs, argnums = node.package
			for argnum, parent in zip(argnums, node.parents):
				if isinstance(parent, anode.NoGradNode):
					# NoGradNode has no _grad slot to write into, and
					# conceptually shouldn't receive gradient anyway - without
					# this check, ANY graph where a no-grad value feeds into
					# an otherwise-differentiable computation (`mask = x > 0;
					# y = x * mask` is enough) crashed here.
					continue
				grad_fn = node.grad_fns[argnum]
				# Gradient errors must propagate immediately. Suppressing them here
				# allowed a stale parent_grad from an earlier operation to be reused,
				# silently producing incorrect gradients during training.
				parent_grad = grad_fn(value, g, *args, **kwargs)
				outgrads[parent] = utils.sum_with_none(outgrads.get(parent), parent_grad)
				parent._grad = outgrads[parent]
		return g

	@classmethod
	def register(cls, value_type):
		Variable.types.add(cls)
		Variable.type_mappings[value_type] = cls
