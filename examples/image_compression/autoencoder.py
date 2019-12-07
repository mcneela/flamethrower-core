from __future__ import division

import numpy as np

import flameflower.nn as nn
import flameflower.nn.activations as act
import flameflower.optim as optim
import flameflower.autograd.tensor_library as tl


class Encoder(nn.Module):
	def __init__(self, num_in, hidden_dim=32):
		super(Encoder, self).__init__()
		self.num_in = num_in
		self.hidden_dim = hidden_dim
		self.lin1 = nn.Linear(num_in, 128)
		self.lin2 = nn.Linear(128, 64)
		self.lin3 = nn.Linear(64, hidden_dim)
		self.add_module('linear1', self.lin1)
		self.add_module('linear2', self.lin2)
		self.add_module('linear3', self.lin3)

	def forward(self, X):
		out = act.relu(self.lin1(X))
		out = act.relu(self.lin2(out))
		out = act.relu(self.lin3(out))
		return out

class Decoder(nn.Module):
	def __init__(self, num_out, hidden_dim=32):
		super(Decoder, self).__init__()
		self.hidden_dim = hidden_dim
		self.num_out = num_out
		self.lin1 = nn.Linear(hidden_dim, 64)
		self.lin2 = nn.Linear(64, 128)
		self.lin3 = nn.Linear(128, 784)
		self.add_module('linear1', self.lin1)
		self.add_module('linear2', self.lin2)
		self.add_module('linear3', self.lin3)

	def forward(self, X):
		out = act.relu(self.lin1(X))
		out = act.relu(self.lin2(out))
		out = act.sigmoid(self.lin3(out))
		return out

class Autoencoder(nn.Module):
	def __init__(self, num_in, num_out, hidden_dim=32):
		super(Autoencoder, self).__init__()
		self.num_in = num_in
		self.num_out = num_out
		self.hidden_dim = hidden_dim
		self.encoder = Encoder(num_in, hidden_dim=hidden_dim)
		self.decoder = Decoder(num_out, hidden_dim=hidden_dim)
		self.add_module('encoder', self.encoder)
		self.add_module('decoder', self.decoder)

	def forward(self, X):
		encoded = self.encoder(X)
		decoded = self.decoder(encoded)
		return decoded


