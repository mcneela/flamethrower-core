"""
Trains a small two-layer network to learn XOR.

XOR is the classic toy problem for this purpose: it is not linearly
separable, so a single Linear layer cannot solve it - the network needs a
hidden layer and a nonlinearity in between. That makes it a good smoke test
for the library as a whole: Module/Linear composition, an activation, a
loss, backward(), and an optimizer step all have to work together correctly
for this to converge.
"""
import numpy as np

import flamethrower.nn as nn
import flamethrower.nn.activations as F
import flamethrower.nn.loss as loss_fns
import flamethrower.optim as optim
from flamethrower.autograd import Tensor

np.random.seed(0)


class XORNet(nn.Module):
	def __init__(self):
		super(XORNet, self).__init__()
		self.hidden = nn.Linear(2, 8)
		self.output = nn.Linear(8, 1)
		self.add_module('hidden', self.hidden)
		self.add_module('output', self.output)

	def forward(self, X):
		h = F.tanh(self.hidden(X))
		return F.sigmoid(self.output(h))


X = Tensor(np.array([
	[0.0, 0.0],
	[0.0, 1.0],
	[1.0, 0.0],
	[1.0, 1.0],
]))
y = np.array([[0.0], [1.0], [1.0], [0.0]])

model = XORNet()
optimizer = optim.SGD(model.params(), lr=1.0)

num_epochs = 3000
for epoch in range(num_epochs):
	optimizer.zero_grad()
	predictions = model(X)
	loss = loss_fns.binary_cross_entropy(y, predictions)
	loss.backward()
	optimizer.step()

	if epoch % 500 == 0 or epoch == num_epochs - 1:
		print("epoch {:4d}  loss {:.4f}".format(epoch, float(loss.data)))

print("\nFinal predictions vs targets:")
final = model(X)
for inputs, pred, target in zip(X.data, final.data, y):
	print("  {} -> {:.3f}  (target {:.0f})".format(inputs, pred[0], target[0]))

predicted_labels = (final.data > 0.5).astype(float)
accuracy = (predicted_labels == y).mean()
print("\naccuracy: {:.0%}".format(accuracy))
assert accuracy == 1.0, "XOR should be solved exactly by a converged network this small."
