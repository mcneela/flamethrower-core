import flameflower.autograd.tensor_library as tl
from .module import Module

class BatchNorm(Module):
	def __init__(self, eps=1e-5, gamma=1, beta=0):
		super(BatchNorm, self).__init__()
		self.eps = eps
		self.gamma = gamma
		self.beta = beta

	def forward(self, X):
		n = X.data.size()[0]
		mu = tl.sum(X, axis=1) / n
		std = tl.sum((X - mu) ** 2) / n
		x_hat = (X - mu) / tl.sqrt(std + self.eps)
		y = self.gamma * x_hat + self.beta
		return y
