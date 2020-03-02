from __future__ import division

import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class FeedforwardNetwork(nn.Module):
	def __init__(self, num_in, num_out):
		super(FeedforwardNetwork, self).__init__()
		self.num_in = num_in
		self.num_out = num_out
		self.lin1 = nn.Linear(num_in, 200)
		self.lin2 = nn.Linear(200, 60)
		self.lin3 = nn.Linear(60, num_out)

	def forward(self, X):
		out = F.relu(self.lin1(X))
		out = F.relu(self.lin2(out))
		out = self.lin3(out)
		return out
