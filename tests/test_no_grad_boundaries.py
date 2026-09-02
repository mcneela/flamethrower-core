import numpy as np

from flamethrower.autograd import Tensor


def test_boolean_mask_from_comparison_can_be_used_in_arithmetic():
    # A very natural pattern - masking with a boolean comparison rather than
    # tl.where - used to crash backward() with a raw AttributeError, because
    # the comparison's NoGradNode has no _grad slot to write into and no
    # grad_fns to consult. Gradient should simply not flow through the mask,
    # the same as PyTorch's detach()/stop_gradient.
    x = Tensor(np.array([-2.0, 3.0, -1.0, 5.0]))
    mask = x > 0
    y = x * mask
    y.backward()

    np.testing.assert_allclose(x.grad, [0.0, 1.0, 0.0, 1.0])


def test_backward_on_a_pure_comparison_leaves_its_input_gradient_as_none():
    x = Tensor(np.array([1.0, 2.0]))
    mask = x > 0

    mask.backward()

    assert x.grad is None


def test_fresh_tensor_grad_is_none_instead_of_raising():
    x = Tensor(np.array([1.0, 2.0]))
    assert x.grad is None
