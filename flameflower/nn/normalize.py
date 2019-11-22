import flameflower.autograd.tensor_library as tl
from flameflower.autograd import Tensor
from .module import Module

class BatchNorm(Module):
	def __init__(self, eps=1e-5, gamma=1, beta=0):
		super(BatchNorm, self).__init__()
		self.eps = eps
		self.gamma = Tensor(gamma)
		self.beta = Tensor(beta)

	def forward(self, X):
		n = X.shape[0]
		mu = tl.sum(X, axis=0) / n
		print(f"Mu is: {mu.data}")
		X - mu
		std = tl.sum((X - mu) ** 2, axis=0) / n
		x_hat = (X - mu) / tl.sqrt(std + self.eps)
		y = self.gamma * x_hat + self.beta
		print(y.shape)
		return y
