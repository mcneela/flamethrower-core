import flamethrower.autograd.tensor_library as tl

def sigmoid(x):
	return 1 / (1 + tl.exp(-x))

def softmax(x, axis=1):
	x = x - tl.max(x, axis=axis, keepdims=True)
	y = tl.exp(x)
	return y / tl.sum(y, axis=axis, keepdims=True)

def relu(x):
	return tl.maximum(0, x)

def elu(x, alpha=1.0):
	return relu(x) + tl.minimum(0, alpha * (tl.exp(x) - 1))

def hardshrink(x, lval=0.5):
	z = tl.copy(x)
	z[(x >= -lval) & (x <= lval)] = 0
	return z

def leaky_relu(x, alpha=0.1):
	return relu(x) + alpha * tl.minimum(0, x)

def relu6(x):
	return tl.minimum(tl.maximum(0, x), 6)

def rrelu(x, lo=0.125, hi=1/3):
	# A plain `if x >= 0` only works for a single-element x: comparing an
	# array yields an array, and Python can't branch on that. tl.where
	# applies the per-element choice instead. The condition must stay a raw
	# array (x.data), not a traced Tensor comparison, or its no-grad node
	# would end up as a parent of this differentiable node during backward().
	noise = tl.random.uniform(lo, hi, size=x.shape)
	return tl.where(x.data >= 0, x, noise * x)

def selu(x, alpha=1.6732632423543772848170429916717):
	return 1.0507009873554804934193349852946 * elu(x, alpha=alpha)

def celu(x, alpha=1.0):
	return relu(x) + tl.minimum(0, alpha * (tl.exp(x / alpha) - 1))

def softplus(x, beta=1, threshold=20):
	stable = beta * x.data <= threshold
	# tl.where evaluates both branches for every element (there's no lazy
	# short-circuiting like Python's `if`), so beta*x must be kept small
	# before it ever reaches exp() - otherwise the overflowed exp(beta*x)
	# in the discarded elements can turn into a NaN gradient once the
	# zeroed-out incoming gradient from tl.where multiplies through it
	# (0 * inf = NaN). Substituting 0 there keeps that branch finite;
	# tl.where still picks `x` itself for those elements in the output.
	safe_x = tl.where(stable, x, tl.zeros(x.shape))
	approx = (1 / beta) * tl.log(1 + tl.exp(beta * safe_x))
	return tl.where(stable, approx, x)

def softsign(x):
	return x / (1 + tl.abs(x))

def tanh(x):
	return tl.tanh(x)

def tanhshrink(x):
	return x - tl.tanh(x)

def threshold(x, val):
	# The original compared x to the `threshold` function object itself
	# (a leftover reference to this function's own name, not `val`), and
	# used a plain Python `if` that would also break on any array with more
	# than one element.
	return tl.where(x.data > val, x, val)

def softmin(x, axis=1):
	x = -x - tl.max(x, axis=axis, keepdims=True)
	y = tl.exp(x)
	return y / tl.sum(y, axis=axis, keepdims=True)

def log_softmax(x):
	return tl.log(softmax(x))

def hardtanh(x, minx=-1, maxx=1, minv=-1, maxv=1):
	z = tl.copy(x)
	z[z > maxx] = maxv
	z[z < minx] = minv
	return z

def softshrink(x, lval=0.5):
	z = tl.zeros(x.shape)
	z[x > lval] = (x - lval)[x > lval]
	z[x < -lval] = (x + lval)[x < -lval]
	return z
