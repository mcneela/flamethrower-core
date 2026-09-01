import numpy as np
import pytest

from flamethrower.autograd import Tensor
from flamethrower.nn import Linear
from flamethrower.optim import SGD
from flamethrower.optim.rprop import RProp


def set_grad(param, value):
	param.node._grad = np.asarray(value, dtype=float)


def test_sgd_accepts_parameter_generator_and_merges_defaults():
	layer = Linear(2, 1)
	optimizer = SGD(layer.params(), lr=0.25)

	assert optimizer.params == [layer.W, layer.b]
	assert optimizer.param_groups[0]['lr'] == 0.25
	assert optimizer.param_groups[0]['use_momentum'] is False


def test_sgd_updates_parameters_and_zero_grad_skips_them():
	param = Tensor(np.array([1.0, -1.0]))
	optimizer = SGD([param], lr=0.1)
	set_grad(param, [2.0, -3.0])

	optimizer.step()
	np.testing.assert_allclose(param.data, [0.8, -0.7])

	optimizer.zero_grad()
	assert param.grad is None
	optimizer.step()
	np.testing.assert_allclose(param.data, [0.8, -0.7])


def test_sgd_momentum_moves_opposite_the_gradient_and_persists_state():
	param = Tensor(np.array([1.0]))
	optimizer = SGD([param], lr=0.1, use_momentum=True, beta=0.9)

	set_grad(param, [2.0])
	optimizer.step()
	set_grad(param, [2.0])
	optimizer.step()

	np.testing.assert_allclose(optimizer.state[param]['v'], [3.8])
	np.testing.assert_allclose(param.data, [0.42])


def test_rprop_adapts_step_size_and_mutates_the_original_parameter():
	param = Tensor(np.array([1.0, -1.0]))
	optimizer = RProp([param], lr=0.1)

	set_grad(param, [1.0, -1.0])
	optimizer.step()
	set_grad(param, [1.0, -1.0])
	optimizer.step()

	np.testing.assert_allclose(optimizer.state[param]['step_size'], [0.12, 0.12])
	np.testing.assert_allclose(param.data, [0.78, -0.78])


def test_rprop_shrinks_step_size_and_skips_update_on_sign_disagreement():
	# Regression test: the sign-remapping used to apply its three boolean masks
	# in place on the same array. Since eta_minus (0.5) is itself positive, the
	# `sign > 0` mask would immediately re-catch and overwrite entries just set
	# to eta_minus, so a gradient sign flip could never shrink the step size -
	# it silently grew every step instead, regardless of sign agreement.
	param = Tensor(np.array([1.0]))
	optimizer = RProp([param], lr=0.1)

	set_grad(param, [2.0])
	optimizer.step()
	np.testing.assert_allclose(optimizer.state[param]['step_size'], [0.1])
	np.testing.assert_allclose(param.data, [0.9])

	set_grad(param, [-3.0])  # sign flips relative to the previous gradient
	optimizer.step()

	# Disagreement must shrink the step size by eta_minus (0.5), not grow it.
	np.testing.assert_allclose(optimizer.state[param]['step_size'], [0.05])
	# The update for a disagreeing gradient is zeroed for this step.
	np.testing.assert_allclose(param.data, [0.9])


def test_parameter_group_can_override_default_learning_rate():
	param = Tensor(np.array([1.0]))
	optimizer = SGD([{'params': [param], 'lr': 0.25}], lr=0.1)
	set_grad(param, [1.0])

	optimizer.step()

	np.testing.assert_allclose(param.data, [0.75])


def test_optimizer_rejects_a_duplicate_parameter_within_one_group():
	param = Tensor(np.array([1.0]))

	with pytest.raises(ValueError, match="more than once"):
		SGD([param, param], lr=0.1)


@pytest.mark.parametrize(
	"kwargs",
	[
		{'lr': 0},
		{'step_sizes': (0, 1)},
		{'step_sizes': (2, 1)},
	],
)
def test_rprop_rejects_invalid_step_configuration(kwargs):
	param = Tensor(np.array([1.0]))

	with pytest.raises(ValueError):
		RProp([param], **kwargs)
