import numpy as np
import pytest

from flamethrower.autograd import Tensor
import flamethrower.autograd.tensor_library as tl
import flamethrower.nn.activations as activations


def test_sigmoid_is_finite_for_extreme_values_and_has_correct_zero_gradient():
    x = Tensor(np.array([-1000.0, 0.0, 1000.0]))

    output = activations.sigmoid(x)
    tl.sum(output).backward()

    np.testing.assert_allclose(output.data, [0.0, 0.5, 1.0], atol=1e-15)
    np.testing.assert_allclose(x.grad, [0.0, 0.25, 0.0], atol=1e-15)
    assert np.all(np.isfinite(output.data))
    assert np.all(np.isfinite(x.grad))


@pytest.mark.parametrize(
    "activation, expected, expected_grad",
    [
        (
            lambda x: activations.hardshrink(x, lval=0.5),
            [-1.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0, 1.0],
        ),
        (
            lambda x: activations.softshrink(x, lval=0.5),
            [-0.5, 0.0, 0.0, 0.0, 0.5],
            [1.0, 0.0, 0.0, 0.0, 1.0],
        ),
    ],
)
def test_shrink_activations_are_functional_and_differentiable(
    activation, expected, expected_grad
):
    x = Tensor(np.array([-1.0, -0.5, 0.0, 0.5, 1.0]))

    output = activation(x)
    tl.sum(output).backward()

    np.testing.assert_allclose(output.data, expected)
    np.testing.assert_allclose(x.grad, expected_grad)


def test_hardtanh_clips_values_without_in_place_tensor_assignment():
    x = Tensor(np.array([-2.0, -0.5, 0.0, 0.5, 2.0]))

    output = activations.hardtanh(x)
    tl.sum(output).backward()

    np.testing.assert_allclose(output.data, [-1.0, -0.5, 0.0, 0.5, 1.0])
    np.testing.assert_allclose(x.grad, [0.0, 1.0, 1.0, 1.0, 0.0])


def test_threshold_has_separate_cutoff_and_replacement_values():
    x = Tensor(np.array([-2.0, 0.0, 3.0]))

    output = activations.threshold(x, threshold_value=1.0, value=-0.5)
    tl.sum(output).backward()

    np.testing.assert_allclose(output.data, [-0.5, -0.5, 3.0])
    np.testing.assert_allclose(x.grad, [0.0, 0.0, 1.0])


def test_softmin_and_log_softmax_are_stable_and_use_the_last_axis_by_default():
    logits = Tensor(np.array([-1000.0, 1000.0]))

    softmin = activations.softmin(logits)
    log_softmax = activations.log_softmax(logits)

    np.testing.assert_allclose(softmin.data, [1.0, 0.0], atol=1e-15)
    np.testing.assert_allclose(tl.sum(softmin).data, 1.0)
    np.testing.assert_allclose(log_softmax.data, [-2000.0, 0.0])
    assert np.all(np.isfinite(softmin.data))
    assert np.all(np.isfinite(log_softmax.data))

    tl.sum(log_softmax).backward()
    np.testing.assert_allclose(logits.grad, [1.0, -1.0])


def test_softplus_remains_finite_for_extreme_values():
    x = Tensor(np.array([-1000.0, 0.0, 1000.0]))

    output = activations.softplus(x)
    tl.sum(output).backward()

    np.testing.assert_allclose(output.data, [0.0, np.log(2), 1000.0], atol=1e-12)
    np.testing.assert_allclose(x.grad, [0.0, 0.5, 1.0], atol=1e-12)
    assert np.all(np.isfinite(x.grad))


@pytest.mark.parametrize(
    "call",
    [
        lambda x: activations.hardshrink(x, lval=-1),
        lambda x: activations.softshrink(x, lval=-1),
        lambda x: activations.hardtanh(x, minx=2, maxx=1),
        lambda x: activations.softplus(x, beta=0),
        lambda x: activations.celu(x, alpha=0),
        lambda x: activations.rrelu(x, lo=0.5, hi=0.25),
    ],
)
def test_activation_parameter_validation(call):
    with pytest.raises(ValueError):
        call(Tensor(np.array([1.0])))
