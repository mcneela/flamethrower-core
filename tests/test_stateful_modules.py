import numpy as np
import pytest

from flamethrower.autograd import Tensor
import flamethrower.autograd.tensor_library as tl
from flamethrower.nn import BatchNorm, Dropout, Module
import flamethrower.nn.normalize as normalize
from flamethrower.optim import SGD


class ChildContainer(Module):
	def __init__(self):
		super(ChildContainer, self).__init__()
		self.dropout = Dropout(p=0.25)
		self.add_module('dropout', self.dropout)


class ParentContainer(Module):
	def __init__(self):
		super(ParentContainer, self).__init__()
		self.child = ChildContainer()
		self.batch_norm = BatchNorm(num_features=2)
		self.add_module('child', self.child)
		self.add_module('batch_norm', self.batch_norm)


def test_children_and_modes_use_the_registered_module_collection():
	model = ParentContainer()

	assert list(model.children()) == [model.child, model.batch_norm]
	model.eval()
	assert not model.is_training
	assert not model.child.is_training
	assert not model.child.dropout.is_training
	assert not model.batch_norm.is_training

	# The original API remains a compatibility alias for the same state path.
	model.set_train_mode()
	assert model.is_training
	assert model.child.is_training
	assert model.child.dropout.is_training
	assert model.batch_norm.is_training

	with pytest.raises(TypeError, match="boolean"):
		model.train(1)


def test_dropout_uses_drop_probability_and_inverted_scaling():
	p = 0.25
	values = np.ones(8)

	# NumPy and tl.random share the same generator. Re-seeding gives an exact,
	# deterministic expected mask without relying on statistical tolerances.
	np.random.seed(7)
	expected_mask = np.random.uniform(0, 1, size=values.shape) >= p
	np.random.seed(7)

	x = Tensor(values.copy())
	dropout = Dropout(p=p)
	output = dropout(x)
	expected = expected_mask / (1 - p)
	np.testing.assert_allclose(output.data, expected)

	tl.sum(output).backward()
	np.testing.assert_allclose(x.grad, expected)

	dropout.eval()
	assert dropout(x) is x
	assert dropout.on is False
	dropout.train_mode()
	assert dropout.on is True


def test_dropout_boundary_probabilities_and_validation():
	x = Tensor(np.ones(3))
	assert Dropout(p=0)(x) is x

	dropped = Dropout(p=1)(x)
	np.testing.assert_allclose(dropped.data, np.zeros(3))

	with pytest.raises(ValueError, match="between zero and one"):
		Dropout(p=-0.1)
	with pytest.raises(ValueError, match="between zero and one"):
		Dropout(p=1.1)


def test_batch_norm_has_one_canonical_class_and_registered_state():
	assert normalize.BatchNorm is BatchNorm

	layer = BatchNorm(num_features=3)
	assert [name for name, _ in layer.named_parameters()] == ['gamma', 'beta']
	assert list(layer._buffers) == [
		'running_mean',
		'running_var',
		'num_batches_tracked',
	]
	assert layer.running_mean.is_tracked is False
	assert layer.running_var.is_tracked is False


def test_batch_norm_training_updates_buffers_and_normalizes_features():
	layer = BatchNorm(num_features=2, momentum=0.5, eps=1e-5)
	x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))

	output = layer(x)

	np.testing.assert_allclose(np.mean(output.data, axis=0), [0.0, 0.0], atol=1e-12)
	np.testing.assert_allclose(np.var(output.data, axis=0), [1.0, 1.0], rtol=1e-5)
	np.testing.assert_allclose(layer.running_mean.data, [1.5, 2.0])
	np.testing.assert_allclose(layer.running_var.data, [11 / 6, 11 / 6])
	assert layer.num_batches_tracked.data == 1

	tl.sum(output * output).backward()
	assert x.grad.shape == x.shape
	assert layer.gamma.grad.shape == (2,)
	assert layer.beta.grad.shape == (2,)


def test_batch_norm_inference_uses_running_statistics_without_updating_them():
	layer = BatchNorm(num_features=2, momentum=0.5, eps=1e-5)
	layer(Tensor(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])))
	running_mean = layer.running_mean.data.copy()
	running_var = layer.running_var.data.copy()
	tracked_batches = layer.num_batches_tracked.data.copy()

	layer.eval()
	x = Tensor(np.array([[10.0, 20.0]]))
	output = layer(x)
	expected = (x.data - running_mean) / np.sqrt(running_var + layer.eps)

	np.testing.assert_allclose(output.data, expected)
	np.testing.assert_allclose(layer.running_mean.data, running_mean)
	np.testing.assert_allclose(layer.running_var.data, running_var)
	assert layer.num_batches_tracked.data == tracked_batches


def test_lazy_batch_norm_preserves_optimizer_parameter_references():
	layer = BatchNorm()
	optimizer = SGD(layer.params(), lr=0.1)
	original_params = list(optimizer.params)

	layer(Tensor(np.arange(12.0).reshape(4, 3)))

	assert optimizer.params == original_params
	assert optimizer.params[0] is layer.gamma
	assert optimizer.params[1] is layer.beta
	assert layer.gamma.shape == (3,)
	assert layer.beta.shape == (3,)

	with pytest.raises(ValueError, match="Expected 3 features"):
		layer(Tensor(np.ones((2, 4))))
