from autograd import grad as gd, jacobian
from tensor_library import *
from core import *

# fn = lambda x: 2 * x + 3 / x
def fn(x):
	z = 3 / x
	y = 2 * x + z
	q = z * y
	return q

hand_grad = lambda x: 2 - 3 / (x ** 2)
start_node = GradNode.new_root()
my_grad = grad(fn)
their_grad = gd(fn)
print("Calculated f'(3) = {}".format(my_grad(4.0)))
# print("Expected   f'(3) = {}".format(hand_grad(4.0)))
print("Their      f'(3) = {}".format(their_grad(4.0)))