# gamma.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import torch

# Define some useful constants
NEGINF = float("-inf")
INF = float("inf")

class Gamma(torch.nn.Module):
	"""A gamma distribution object.

	This distribution represents the sum of several exponential distributions,
	with shape and rate parameters. It assumes that each feature is independent
	of the others, so no covariance is learned.

	Parameters
	----------
	shapes: torch.tensor or None, shape=(d,), optional
		The shape parameter for each feature. Default is None

	rates: torch.tensor or None, shape=(d,), optional
		The rate parameters for each feature. Default is None.

	frozen: bool, optional
		Indicates if the parameters for this distribution should be fixed
		during training. Default is False.

	inertia: float, (0, 1), optional
		Indicates the proportion of the update to apply to the parameters
		during training. When the inertia is 0.0, the update is applied in
		its entirety and the previous parameters are ignored. When the
		inertia is 1.0, the update is entirely ignored and the previous
		parameters are kept, equivalently to if the parameters were frozen.
	"""

	def __init__(self, shapes=None, rates=None, frozen=False, inertia=0.0):
		self.d = len(shapes)
		self.shapes = shapes
		self.rates = torch.tensor(rates)

		self.name = "Gamma"
		self.frozen = frozen
		self.inertia = inertia

		self._initialized = shapes is not None
		self._reset_cache()

	def __reduce__(self):
		return self.__class__, (self.shapes, self.rates, self.frozen, self.inertia)

	def _initialize(self, d):
		self.shapes = torch.zeros(d)
		self.rates = torch.zeros(d)
		self._reset_cache()

	def _reset_cache(self):
		self._w_sum = torch.zeros(self.d)
		self._xw_sum = torch.zeros(self.d)
		self._logx_w_sum = torch.zeros(self.d)

		if self._initialized == True:
			self._log_rates = torch.log(self.rates)
			self._lgamma_shapes = torch.lgamma(self.shapes)
			self._thetas = self._log_rates * self.shapes.data - self._lgamma_shapes

	def log_probability(self, X):
		return torch.sum(self._thetas + torch.log(X) * (self.shapes - 1) - self.rates * X, dim=-1)

	def summarize(self, X, sample_weights=None):
		if not self._initialized:
			self._initialize(X.shape[1])

		if sample_weights is None:
			sample_weights = torch.ones(X.shape[0], 1)

		self._w_sum += torch.sum(sample_weights, axis=(0, 1))
		self._xw_sum += torch.sum(X * sample_weights, dim=0)
		self._logx_w_sum += torch.sum(torch.log(X) * sample_weights, dim=0)

	def from_summaries(self):
		if self._w_sum.sum() < 1e-8 or self.frozen == True:
			return

		epsilon = 1e-4
		max_iterations = 20

		thetas = torch.log(self._xw_sum / self._w_sum) - \
			self._logx_w_sum / self._w_sum

		new_shapes = (3 - thetas + torch.sqrt((thetas - 3) ** 2 + 24 * thetas)) / (12 * thetas)
		shapes = new_shapes + epsilon

		for iteration in range(max_iterations):
			mask = torch.abs(shapes - new_shapes) < epsilon
			if torch.all(mask):
				break

			shapes = new_shapes
			new_shapes = shapes - (torch.log(shapes) - torch.polygamma(0, shapes) - thetas) / (1.0 / shapes - torch.polygamma(1, shapes))


		shapes = new_shapes

		# Now our iterative estimation of the shape parameter has converged.
		# Calculate the rate parameter
		rates = 1.0 / (1.0 / (shapes * self._w_sum) * self._xw_sum)

		# Calculate the new parameters, respecting inertia, with an inertia
		# of 0 being completely replacing the parameters, and an inertia of
		# 1 being to ignore new training data.
		self.shapes = shapes
		self.rates = rates
		self._reset_cache()


class ToyNet(torch.nn.Module):
	def __init__(self, d):
		super(ToyNet, self).__init__()

		self.fc1 = torch.nn.Linear(d, 32)
		self.shapes = torch.nn.Linear(32, d)
		self.rates = torch.nn.Linear(32, d)
		self.relu = torch.nn.ReLU()

	def forward(self, X):
		X = self.fc1(X)
		X = self.relu(X)
		
		shapes = self.shapes(X)
		rates = self.rates(X)
		return self.relu(shapes) + 0.01, self.relu(rates) + 0.01





import numpy
import time 

n = 10000
d = 12

X = torch.randn(n, d)
X = torch.exp(X)

model = ToyNet(d)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for i in range(100):
	optimizer.zero_grad()
	
	shapes, rates = model(X)
	loss = -Gamma(shapes, 2).log_probability(X).sum()

	print(loss.item())
	loss.backward()
	optimizer.step()
