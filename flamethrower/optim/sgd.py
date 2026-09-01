from .optimizer import Optimizer

import logging
import flamethrower.autograd.tensor_library as tl

logger = logging.getLogger(__name__)

class SGD(Optimizer):
	"""
	Implements Stochastic Gradient Descent with and without momentum.
	"""
	def __init__(self, params, lr=1e-3, use_momentum=False,
				 beta=0, lr_scheduler=None):

		# Perform value checking on the keyword arguments/
		if lr < 0.0:
			logger.error("Invalid learning rate: {} - should be >= 0.0".format(lr))
			raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
		if use_momentum and beta <= 0:
			logger.error("Using momentum, but invalid beta value: {} - should be > 0.0".format(beta))
			raise ValueError("Using momentum, but invalid momentum value: {} - should be > 0.0".format(beta))
		if beta < 0.0:
			logger.error("Invalid beta value: {} - should be >= 0.0".format(beta))
			raise ValueError("Invalid momentum value: {} - should be >= 0.0".format(beta))
		# Set Optimizer defaults and __init__ base with these.
		defaults = dict(lr=lr, use_momentum=use_momentum,
						beta=beta, lr_scheduler=lr_scheduler)
		super(SGD, self).__init__(params, defaults)

	def __name__(self):
		return "stochastic_gradient_descent"

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
			# Options are read from the group so callers can use different learning
			# rates or momentum settings for different parts of a model.
			lr = group['lr']
			use_momentum = group['use_momentum']
			beta = group['beta']
			lr_scheduler = group['lr_scheduler']
			if lr_scheduler:
				lr = lr_scheduler.step()
				group['lr'] = lr

			for p in group['params']:
				# A newly created GradNode has no _grad attribute until backward runs.
				# getattr handles both that case and gradients cleared by zero_grad().
				grad = getattr(p.node, '_grad', None)
				if grad is None:
					continue

				if use_momentum:
					# Optimizer state is lazy because parameters that never receive a
					# gradient should not allocate a momentum buffer.
					param_state = self.state.setdefault(p, {})
					if 'v' not in param_state:
						v = tl.zeros_like(grad)
					else:
						v = param_state['v']
					# Momentum accumulates gradients; the learning rate is applied once
					# during the descent step. The previous implementation added a
					# positive-gradient velocity and therefore moved parameters uphill.
					v = beta * v + grad
					param_state['v'] = v
					p.data -= lr * v
				else:
					p.data -= lr * grad

		return loss 
