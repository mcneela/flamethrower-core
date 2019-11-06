from __future__ import absolute_import

class Variable(object):
	type_mappings = {}
	types = set()

	__slots__ = ['_data', '_node']

	def __init__(self, data, node):
		self._data = data
		self._node  = node

	def __bool__(self):
		return bool(self._data)

	__nonzero__ = __bool__

	@classmethod
	def register(cls, value_type):
		Variable.types.add(cls)
		Variable.type_mappings[value_type] = cls

