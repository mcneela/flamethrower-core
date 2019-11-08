from .optimizer import Optimizer
import flameflower.autograd.tensor_library as tl

class SGD(Optimizer):
	"""
	Implements Stochastic Gradient Descent with and without momentum.
	"""
	def __init__(self, params, lr=1e-3, use_momentum=False,
				 beta=0, lr_scheduler=None):

		# Perform value checking on the keyword arguments/
		if lr < 0.0:
			logging.error("Invalid learning rate: {} - should be >= 0.0".format(lr))
			raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
		if use_momentum and beta <= 0:
			logging.error("Using momentum, but invalid beta value: {} - should be >= 0.0".format(lr))
			raise ValueError("Using momentum, but invalid momentum value: {} - should be >= 0.0".format(lr))
		if beta < 0.0:
			logging.error("Invalid beta value: {} - should be >= 0.0".format(lr))
			raise ValueError("Invalid momentum value: {} - should be >= 0.0".format(lr))

		# Set Optimizer defaults and __init__ base with these.
		defaults = dict(lr=lr, use_momentum=use_momentum,
						beta=beta, lr_scheduler=lr_scheduler)
		super(SGD, self).__init__(params, defaults)

	def __setstate__(self, state):
		super(SGD, self).__setstate__(state)

	def step(self, closure=None):
		"""
		Performs a single optimization step.
		"""
		loss = None
		if closure is not None:
			loss = closure()

		for group in self.param_groups:
			lr = group['lr']
			use_momentum = group['use_momentum']
			beta = group['beta']
			lr_scheduler = group['lr_scheduler']
			if lr_scheduler:
				lr = lr_scheduler.step()

			for p in group['params']:
				if p.grad is None:
					continue
				grad = p.grad

				if use_momentum:
					param_state = self.state[p]
					if 'v' not in param_state:
						v = param_state['v'] = tl.copy(grad)
					else:
						v = param_state['v']
					v = beta * v - lr * grad
					param_state['v'] = v
					p.data += v
				else:
					p.data -= lr * grad

		return loss 
