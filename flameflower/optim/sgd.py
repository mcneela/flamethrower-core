from .optimizer import Optimizer

class SGD(Optimizer):
	def __init__(self, params, lr=1e-3, momentum=0, dampening=0, 
				 weight_decay=0, nesterov=False, lr_scheduler=None):
		defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
						weight_decay=weight_decay, nesterov=nesterov)
		super(SGD, self).__init__(params, defaults)
		self.scheduler = lr_scheduler
		self.lr = lr
		if lr < 0.0:
			logging.error("Invalid learning rate: {} - should be >= 0.0".format(lr))
			raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))

	def step(self, iter=1, closure=None):
		for group in self.param_groups:
			for p in group['params']:
				if p.node().grad is None:
					continue
				grad = p.node().grad

				# state = self.state[p]

				# if len(state) == 0:
				# 	state['step'] = 0

				# lr = group['lr']
				# state['step'] += 1
				# print(f"Node: {p.node().name}, Grad: {grad}")

				p._data -= (self.lr/iter) * grad

		return None
