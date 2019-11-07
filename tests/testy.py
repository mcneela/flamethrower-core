import numpy as np
from autograd import grad as gd, jacobian
import tensor_library as tl
from node import *
from tensor import Tensor
from core import *

# fn = lambda x: 2 * x + 3 / x
def fn(x):
	z = 3 / x
	y = 2 * x + z
	# print(z.data().shape)
	# print(y.data().shape)
	q = z.T @ y
	return q

hand_grad = lambda x: 2 - 3 / (x ** 2)
# start_node = GradNode.new_root()
# my_grad = grad(fn)
their_grad = jacobian(fn)
x = Tensor(tl.array([[4.0, 2.0, 3.0], [1.0, 7.0, 6.0]]))
x2 = np.array([[4.0, 2.0, 3.0], [1.0, 7.0, 6.0]])
y = fn(x)

ten_grad = y.backward()
# print("Calculated f'(3) = {}".format(my_grad(4.0)))
# print("Expected   f'(3) = {}".format(hand_grad(4.0)))
print("Their      f'(3) = {}".format(their_grad(x2)))
