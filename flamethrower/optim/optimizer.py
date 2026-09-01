import flamethrower.autograd as ag
import flamethrower.autograd.node as anode
import logging

class Optimizer(object):
	"""
	Abstract base class for neural network optimizers.
	"""
	def __init__(self, params, defaults, name=None):
		self.params = []
		self.defaults = defaults.copy()
		self.name = name
		self.state = {}
		self.param_groups = []

		# A Tensor is itself iterable, so accepting one here would silently turn
		# its elements into separate parameters. Requiring an outer iterable keeps
		# optimizer ownership explicit and prevents updates to temporary slices.
		if isinstance(params, ag.Tensor):
			raise TypeError("Params should be an iterable of Tensors, not a single Tensor.")
		try:
			# Materializing once is important because Module.params() returns a
			# generator. It also lets us validate the complete parameter collection.
			param_groups = list(params)
		except TypeError:
			logging.error("Trying to initialize params with a non-iterable object.")
			raise TypeError("Params should be an iterable of Tensors.")

		if len(param_groups) == 0:
			raise ValueError("Empty parameter list.")
		if not isinstance(param_groups[0], dict):
			param_groups = [{'params': param_groups}]

		for pg in param_groups:
			self.add_param_group(pg)

	def add_param_group(self, param_group):
		r"""Add a param group to the :class:`Optimizer` s `param_groups`.

		This can be useful when fine tuning a pre-trained network as frozen layers can be made
		trainable and added to the :class:`Optimizer` as training progresses.

		Arguments:
			param_group (dict): Specifies what Tensors should be optimized along with group
			specific optimization options.
		"""
		if not isinstance(param_group, dict):
			raise TypeError("A parameter group must be a dictionary.")
		if 'params' not in param_group:
			raise KeyError("A parameter group must define a 'params' entry.")

		# Copy the caller's dictionary so adding defaults below does not mutate
		# user-owned configuration unexpectedly.
		param_group = param_group.copy()
		group_params = param_group['params']
		if isinstance(group_params, ag.Tensor):
			group_params = [group_params]
		else:
			try:
				group_params = list(group_params)
			except TypeError:
				raise TypeError("The 'params' entry must be a Tensor or an iterable of Tensors.")

		if len(group_params) == 0:
			raise ValueError("A parameter group cannot be empty.")
		for param in group_params:
			if not isinstance(param, ag.Tensor):
				raise TypeError("Optimizer parameters must be Tensors, got {}.".format(type(param)))
		group_param_ids = [id(param) for param in group_params]
		if len(group_param_ids) != len(set(group_param_ids)):
			# A duplicate inside one group is just as dangerous as a duplicate
			# across groups: step() would update the same Tensor more than once.
			raise ValueError("A Tensor cannot appear more than once in a parameter group.")

		# Each group needs a complete option set. Without merging defaults, RProp
		# and custom groups fail later when they try to read values such as 'lr'.
		for option, value in self.defaults.items():
			param_group.setdefault(option, value)
		param_group['params'] = group_params

		# Updating the same Tensor through two groups would apply two steps with
		# potentially different settings, which is almost always an accidental bug.
		existing_params = {id(param) for group in self.param_groups for param in group['params']}
		if any(id(param) in existing_params for param in group_params):
			raise ValueError("A Tensor cannot appear in more than one parameter group.")

		self.param_groups.append(param_group)
		self.params.extend(group_params)

	def zero_grad(self):
		"""Clear gradients for every managed parameter."""
		for group in self.param_groups:
			for param in group['params']:
				# Gradients live on GradNodes rather than on Tensor.data. Setting them
				# to None makes a missing backward pass distinguishable from a true
				# all-zero gradient and allows step() to skip that parameter safely.
				if isinstance(param.node, anode.GradNode):
					param.node._grad = None

	def __getstate__(self):
		return {
			'defaults': self.defaults,
			'state': self.state,
			'params': self.param_groups
		}

	def __setstate__(self, state):
		logging.info(f"Setting state: {state}")
		self.__dict__.update(state)

	def step(self, closure=None):
		raise NotImplementedError
