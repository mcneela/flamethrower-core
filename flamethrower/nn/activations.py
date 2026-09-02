import flamethrower.autograd.tensor_library as tl

def sigmoid(x):
	# sigmoid(x) = exp(-softplus(-x)) is mathematically identical to
	# 1/(1+exp(-x)) but never overflows: softplus(z) >= 0 for every z, so the
	# argument to exp() here is always <= 0. It also reuses softplus's own
	# overflow guard instead of duplicating that logic.
	return tl.exp(-softplus(-x))

def softmax(x, axis=-1):
	x = x - tl.max(x, axis=axis, keepdims=True)
	y = tl.exp(x)
	return y / tl.sum(y, axis=axis, keepdims=True)

def relu(x):
	return tl.maximum(0, x)

def elu(x, alpha=1.0):
	return relu(x) + tl.minimum(0, alpha * (tl.exp(x) - 1))

def hardshrink(x, lval=0.5):
	if lval < 0:
		raise ValueError("hardshrink requires lval >= 0.")
	# tl.copy(x) plus boolean-mask __setitem__ mutates x's data directly and
	# isn't tracked by autograd, so gradients through it would be wrong (the
	# masked-out elements would incorrectly still show a gradient of 1, as if
	# they'd been copied unchanged). tl.where builds the selection as part of
	# the traced graph instead, so it's actually differentiable.
	in_band = (x.data >= -lval) & (x.data <= lval)
	return tl.where(in_band, tl.zeros(x.shape), x)

def leaky_relu(x, alpha=0.1):
	return relu(x) + alpha * tl.minimum(0, x)

def relu6(x):
	return tl.minimum(tl.maximum(0, x), 6)

def rrelu(x, lo=0.125, hi=1/3):
	if lo > hi:
		raise ValueError("rrelu requires lo <= hi.")
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
	if alpha == 0:
		raise ValueError("celu requires a nonzero alpha (it divides by alpha).")
	return relu(x) + tl.minimum(0, alpha * (tl.exp(x / alpha) - 1))

def softplus(x, beta=1, threshold=20):
	if beta == 0:
		raise ValueError("softplus requires a nonzero beta (it divides by beta).")
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

def softmin(x, axis=-1):
	neg_x = -x
	# The stability shift must be the max of what's actually exponentiated
	# (-x), not of x itself - subtracting max(x) here left large-magnitude
	# negative inputs unprotected from overflow, since negating them produces
	# large positive numbers that the shift wasn't sized for.
	neg_x = neg_x - tl.max(neg_x, axis=axis, keepdims=True)
	y = tl.exp(neg_x)
	return y / tl.sum(y, axis=axis, keepdims=True)

def log_softmax(x, axis=-1):
	# tl.log(softmax(x)) loses precision (or returns -inf) whenever softmax
	# underflows to exactly 0 for very negative logits. Subtracting
	# log-sum-exp directly avoids ever computing that intermediate softmax.
	shifted = x - tl.max(x, axis=axis, keepdims=True)
	return shifted - tl.log(tl.sum(tl.exp(shifted), axis=axis, keepdims=True))

def hardtanh(x, minx=-1, maxx=1, minv=-1, maxv=1):
	if minx > maxx:
		raise ValueError("hardtanh requires minx <= maxx.")
	# See hardshrink() above: tl.where keeps this differentiable, unlike the
	# tl.copy(x) + boolean-mask __setitem__ this used to do.
	clipped = tl.where(x.data > maxx, maxv, x)
	return tl.where(x.data < minx, minv, clipped)

def softshrink(x, lval=0.5):
	if lval < 0:
		raise ValueError("softshrink requires lval >= 0.")
	# tl.zeros(x.shape) returns a raw ndarray (there's no Tensor argument to
	# trace), so indexing it with a Tensor-valued boolean mask - as the
	# original z[x > lval] = ... did - isn't valid numpy indexing. tl.where
	# avoids constructing that intermediate array at all, and is
	# differentiable in the same way hardshrink() and hardtanh() now are.
	positive_branch = tl.where(x.data > lval, x - lval, tl.zeros(x.shape))
	return tl.where(x.data < -lval, x + lval, positive_branch)
