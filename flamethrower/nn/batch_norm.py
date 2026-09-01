import numpy as np

import flamethrower.autograd.tensor_library as tl
from flamethrower.autograd import Tensor
from .module import Module


class BatchNorm(Module):
	"""Feature-last batch normalization with trainable affine parameters.

	The layer normalizes over every axis except the final feature axis. Running
	statistics are buffers: they persist with the module but are not optimized.
	``num_features`` may be omitted for compatibility, in which case parameter
	and buffer arrays are resized lazily on the first forward pass.
	"""

	def __init__(self, eps=1e-5, gamma=1, beta=0, momentum=0.1,
				 num_features=None):
		super(BatchNorm, self).__init__()
		if eps <= 0:
			raise ValueError("BatchNorm epsilon must be greater than zero.")
		if not 0 <= momentum <= 1:
			raise ValueError("BatchNorm momentum must be between zero and one.")
		if num_features is not None and (not isinstance(num_features, int) or num_features <= 0):
			raise ValueError("num_features must be a positive integer or None.")

		self.eps = eps
		self.momentum = momentum
		self.num_features = num_features

		# Register affine terms in both eager and lazy modes so an optimizer
		# created before the first forward pass still owns these Tensor objects.
		self.gamma = Tensor(self._initial_values(gamma, num_features, "gamma"))
		self.beta = Tensor(self._initial_values(beta, num_features, "beta"))
		self.new_param("gamma", self.gamma)
		self.new_param("beta", self.beta)

		self.running_mean = Tensor(
			np.zeros(num_features) if num_features is not None else np.asarray(0.0),
			track=False,
		)
		self.running_var = Tensor(
			np.ones(num_features) if num_features is not None else np.asarray(1.0),
			track=False,
		)
		self.num_batches_tracked = Tensor(np.asarray(0), track=False)
		self.new_buffer("running_mean", self.running_mean)
		self.new_buffer("running_var", self.running_var)
		self.new_buffer("num_batches_tracked", self.num_batches_tracked)

	@staticmethod
	def _initial_values(value, num_features, name):
		values = np.asarray(value, dtype=float)
		if num_features is None:
			return values.copy()
		try:
			return np.broadcast_to(values, (num_features,)).copy()
		except ValueError:
			raise ValueError("{} must be scalar or have shape ({},).".format(name, num_features))

	def _ensure_feature_size(self, feature_count):
		if self.num_features is not None:
			if feature_count != self.num_features:
				raise ValueError(
					"Expected {} features, received {}.".format(self.num_features, feature_count)
				)
			return

		# Resize data on the registered objects. Replacing the Tensor objects
		# would leave an optimizer holding obsolete parameter references.
		self.gamma.data = self._initial_values(self.gamma.data, feature_count, "gamma")
		self.beta.data = self._initial_values(self.beta.data, feature_count, "beta")
		self.running_mean.data = np.zeros(feature_count)
		self.running_var.data = np.ones(feature_count)
		self.num_features = feature_count

	def forward(self, X):
		if not isinstance(X, Tensor):
			raise TypeError("BatchNorm input must be a Tensor.")
		if X.dims < 2:
			raise ValueError("BatchNorm input must include a batch axis and a feature axis.")

		self._ensure_feature_size(X.shape[-1])
		reduction_axes = tuple(range(X.dims - 1))
		sample_count = int(np.prod(X.shape[:-1]))

		if self.is_training:
			mean = tl.sum(X, axis=reduction_axes) / sample_count
			variance = tl.sum((X - mean) ** 2, axis=reduction_axes) / sample_count

			# Running statistics are state, not part of the gradient graph. Updating
			# them from raw data prevents gradients from leaking across batches.
			self.running_mean.data = (
				(1 - self.momentum) * self.running_mean.data
				+ self.momentum * mean.data
			)
			self.running_var.data = (
				(1 - self.momentum) * self.running_var.data
				+ self.momentum * variance.data
			)
			self.num_batches_tracked.data += 1
		else:
			# Raw buffer arrays are constants during inference; using the untracked
			# Tensor wrappers directly would incorrectly add them to the graph.
			mean = self.running_mean.data
			variance = self.running_var.data

		normalized = (X - mean) / tl.sqrt(variance + self.eps)
		return self.gamma * normalized + self.beta
