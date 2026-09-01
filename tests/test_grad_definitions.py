import numpy as np

from flamethrower.autograd import Tensor
import flamethrower.autograd.tensor_library as tl
from flamethrower.nn.loss import binary_cross_entropy


def test_maximum_and_minimum_gradients_are_elementwise_and_split_ties():
    x = Tensor(np.array([1.0, 2.0, 3.0]))
    y = Tensor(np.array([0.0, 2.0, 4.0]))
    tl.sum(tl.maximum(x, y)).backward()

    np.testing.assert_allclose(x.grad, [1.0, 0.5, 0.0])
    np.testing.assert_allclose(y.grad, [0.0, 0.5, 1.0])

    x = Tensor(np.array([1.0, 2.0, 3.0]))
    y = Tensor(np.array([0.0, 2.0, 4.0]))
    tl.sum(tl.minimum(x, y)).backward()

    np.testing.assert_allclose(x.grad, [0.0, 0.5, 1.0])
    np.testing.assert_allclose(y.grad, [1.0, 0.5, 0.0])


def test_relu_gradient_handles_negative_positive_and_zero_inputs():
    x = Tensor(np.array([-2.0, 0.0, 3.0]))

    tl.sum(tl.maximum(0, x)).backward()

    # maximum uses a symmetric subgradient of 0.5 at an exact tie.
    np.testing.assert_allclose(x.grad, [0.0, 0.5, 1.0])


def test_abs_and_copy_gradients():
    x = Tensor(np.array([-2.0, 0.0, 3.0]))

    tl.sum(tl.copy(tl.abs(x), order='C')).backward()

    np.testing.assert_allclose(x.grad, [-1.0, 0.0, 1.0])


def test_mean_gradient_supports_numpy_positional_argument_order():
    x = Tensor(np.arange(6.0).reshape(2, 3))

    # NumPy's positional order is axis, dtype, out, keepdims.
    tl.sum(tl.mean(x, 0, None, None, True)).backward()

    np.testing.assert_allclose(x.grad, np.full((2, 3), 0.5))


def test_sum_and_mean_where_masks_have_correct_gradients():
    mask = np.array([True, False, True])

    summed = Tensor(np.array([1.0, 2.0, 3.0]))
    tl.sum(summed, where=mask).backward()
    np.testing.assert_allclose(summed.grad, [1.0, 0.0, 1.0])

    averaged = Tensor(np.array([1.0, 2.0, 3.0]))
    tl.mean(averaged, where=mask).backward()
    np.testing.assert_allclose(averaged.grad, [0.5, 0.0, 0.5])


def test_arctan_and_arctanh_gradients_match_their_derivatives():
    arctan_input = Tensor(np.array([-2.0, 0.0, 0.5]))
    tl.sum(tl.arctan(arctan_input)).backward()
    np.testing.assert_allclose(
        arctan_input.grad,
        1 / (1 + arctan_input.data ** 2),
    )

    arctanh_input = Tensor(np.array([-0.5, 0.0, 0.75]))
    tl.sum(tl.arctanh(arctanh_input)).backward()
    np.testing.assert_allclose(
        arctanh_input.grad,
        1 / (1 - arctanh_input.data ** 2),
    )


def test_binary_cross_entropy_can_backpropagate_through_mean():
    eps = 1e-5
    targets = np.array([0.0, 1.0])
    predictions = Tensor(np.array([0.2, 0.8]))

    binary_cross_entropy(targets, predictions, eps=eps).backward()

    expected = np.array([
        1 / (1 - predictions.data[0] + eps),
        -1 / (predictions.data[1] + eps),
    ]) / len(targets)
    np.testing.assert_allclose(predictions.grad, expected)
