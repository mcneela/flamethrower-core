import flameflower.nn as nn
import flameflower.autograd as ag
import flameflower.autograd.utils as utils

from flameflower.autograd import Tensor

def f(x):
	return x ** 2

x = Tensor([1, 2, 3])
y = f(x)

accurate = utils.grad_check(f, x)