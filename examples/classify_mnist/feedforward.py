from __future__ import division

import numpy as np

import flameflower.nn as nn
import flameflower.nn.activations as activate
import flameflower.optim as optim
import flameflower.autograd.tensor_library as tl


class FeedforwardNetwork(nn.Module):
	def __init__(self, num_in, num_out):
		super(FeedforwardNetwork, self).__init__()
		self.num_in = num_in
		self.num_out = num_out
		self.lin1 = nn.Linear(num_in, 128)
		self.lin2 = nn.Linear(128, num_out)
		self.add_module('linear1', self.lin1)
		self.add_module('linear2', self.lin2)

	def forward(self, X):
		out = self.lin1(X)
		out = self.lin2(out)
		return out
