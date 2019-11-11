import flameflower.autograd.tensor_library as tl

def sigmoid(x):
	return 1 / (1 + tl.exp(-x))

def softmax(x):
	return tl.exp(x) / (tl.sum(tl.exp(x)))

def relu(x):
	return tl.max(0, x)

def elu(x, alpha=1.0):
	return relu(x) + tl.min(0, alpha * (tl.exp(x) - 1))

def hardshrink(x, lval=0.5):
	if x > lval or x < -lval:
		return x
	return 0

def leaky_relu(x, alpha=0.1):
	return relu(x) + alpha * tl.min(0, x)

def relu6(x):
	return tl.min(tl.max(0, x), 6)

def rrelu(x, lo=0.125, hi=1/3):
	if x >= 0:
		return x
	return tl.random.uniform(lo, hi) * x

def selu(x, alpha=1.6732632423543772848170429916717):
	return 1.0507009873554804934193349852946 * elu(x, alpha=alpha)

def celu(x, alpha=1.0):
	return relu(x) + tl.min(0, alpha * (tl.exp(x / alpha) - 1))

def softplus(x, beta=1, threshold=20):
	if x <= threshold:
		return (1 / beta) * tl.log(1 + tl.exp(beta * x))
	return x

def softsign(x):
	return x / (1 + tl.abs(x))

def tanh(x):
	return tl.tanh(x)

def tanhshrink(x):
	return x - tl.tanh(x)

def threshold(x, val):
	if x > threshold:
		return x
	return val

def softmin(x):
	return tl.exp(-x) / (tl.sum(tl.exp(-x)))

def log_softmax(x):
	return tl.log(softmax(x))

def hardtanh(x, minx=-1, maxx=1, minv=-1, maxv=1):
	if x > maxx:
		return maxv
	elif x < minx:
		return minv
	else:
		return x

def softshrink(x, lval=0.5):
	if x > lval:
		return x - lval
	elif x < -lval:
		return x + lval
	else:
		return tl.zeros(x.shape)
