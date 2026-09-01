import numpy as np
import pytest

from flamethrower.autograd import Tensor, define_grad
from flamethrower.autograd.call import primitive
import flamethrower.autograd.tensor_library as tl


class DeliberateGradientError(Exception):
    pass


@primitive
def failing_identity(x):
    return x


def raise_gradient_error(ans, grad, x):
    raise DeliberateGradientError("gradient failure must reach the caller")


define_grad(failing_identity, raise_gradient_error)


def test_backward_propagates_gradient_errors_instead_of_reusing_stale_values():
    x = Tensor(np.array([1.0, 2.0]))
    loss = tl.sum(failing_identity(x))

    with pytest.raises(DeliberateGradientError, match="must reach the caller"):
        loss.backward()
