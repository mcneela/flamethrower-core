class _LRScheduler(object):
	"""
	Base class for learning rate schedulers.
	"""
	def __init__(self, starting_lr, iters_per_epoch=None):
		self.curr_iter = 0
		self.iters_per_epoch = iters_per_epoch
		self.curr_epoch = 1
		self.starting_lr = starting_lr
		self.lr = starting_lr

	def step(self):
		raise NotImplementedError

	def reset(self):
		self.curr_iter = 1
		self.curr_epoch = 1
		self.lr = self.starting_lr

class LinearScheduler(_LRScheduler):
	"""
	Implements a linear decay learning rate schedule.
	In other words, lr = lr / n where n is the current
	iteration.
	"""
	def step(self):
		self.curr_iter += 1
		self.lr /= self.curr_iter
		if self.batch_size and (self.curr_iter % self.iters_per_epoch == 0):
			self.epoch += 1

		return self.lr
