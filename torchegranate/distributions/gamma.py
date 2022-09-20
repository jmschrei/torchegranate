# gamma.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import torch

from _utils import _cast_as_tensor
from _utils import _update_parameter

from _distribution import Distribution

def _log(X):
	if X is None:
		return X
	return torch.log(X)

class Gamma(Distribution):
	"""A gamma distribution object.

	A gamma distribution is the sum of exponential distributions, and has shape
	and rate parameters. This distribution assumes that each feature is
	independent of the others. 


	Parameters
	----------
	shapes: torch.tensor or None, shape=(d,), optional
		The shape parameter for each feature. Default is None

	rates: torch.tensor or None, shape=(d,), optional
		The rate parameters for each feature. Default is None.

	inertia: float, (0, 1), optional
		Indicates the proportion of the update to apply to the parameters
		during training. When the inertia is 0.0, the update is applied in
		its entirety and the previous parameters are ignored. When the
		inertia is 1.0, the update is entirely ignored and the previous
		parameters are kept, equivalently to if the parameters were frozen.

	tol: float, [0, inf), optional
		The threshold at which to stop fitting the parameters of the
		distribution. Default is 1e-4.

	max_iter: int, [0, inf), optional
		The maximum number of iterations to run EM when fitting the parameters
		of the distribution. Default is 20.


	Examples
	--------
	>>> # Create a distribution with known parameters
	>>> shapes = torch.tensor([0.2, 0.6])
	>>> rates = torch.tensor([1.2, 0.4])
	>>> X = torch.tensor([[0.3, 0.2], [0.8, 0.1]])
	>>>
	>>> d = Gamma(shapes, rates)
	>>> d.log_probability(X)
	tensor([-1.2687, -2.3361])
	>>>
	>>>
	>>> # Fit a distribution to data
	>>> n, d = 100, 10
	>>> X = torch.exp(torch.randn(d) * 15)
	>>> 
	>>> d = Gamma().fit(X)
	>>>
	>>>
	>>> # Fit a distribution using the summarize API
	>>> n, d = 100, 10
	>>> X = torch.exp(torch.randn(d) * 15)
	>>> 
	>>> d = Gamma()
	>>> d.summarize(X[:50])
	>>> d.summarize(X[50:])
	>>> d.from_summaries()
	>>>
	>>>
	>>> # As a loss function for a neural network
	>>> class ToyNet(torch.nn.Module):
	>>> 	def __init__(self, d):
	>>>			super(ToyNet, self).__init__()
	>>>			self.fc1 = torch.nn.Linear(d, 32)
	>>>			self.shapes = torch.nn.Linear(32, d)
	>>>			self.rates = torch.nn.Linear(32, d)
	>>>			self.relu = torch.nn.ReLU()
	>>>
	>>>		def forward(self, X):
	>>>			X = self.fc1(X)
	>>>			X = self.relu(X)
	>>>			shapes = self.shapes(X)
	>>>			rates = self.rates(X)
	>>>			return self.relu(shapes) + 0.01, self.relu(rates) + 0.01
	>>>
	>>> n, d = 1000
	>>> X = torch.exp(torch.randn(n, d) * 15)
	>>>
	>>> model = ToyNet(d)
	>>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
	>>>
	>>> for i in range(100):
	>>>		optimizer.zero_grad()
	>>>
	>>>		shapes, rates = model(X)
	>>> 	loss = -Gamma(shapes, 2).log_probability(X).sum()
	>>>		loss.backward()
	>>>		optimizer.step()
	"""

	def __init__(self, shapes=None, rates=None, inertia=0.0, tol=1e-4, 
		max_iter=20):
		super().__init__()
		self.name = "Gamma"
		self.inertia = inertia
		self.tol = tol
		self.max_iter = max_iter

		self._shapes = _log(_cast_as_tensor(shapes))
		self._rates = _log(_cast_as_tensor(rates))

		self._initialized = shapes is not None
		self.d = len(self.shapes) if self._initialized else None
		self._reset_cache()

	@property
	def shapes(self):
		return torch.exp(self._shapes)

	@property
	def rates(self):
		return torch.exp(self._rates)

	def _initialize(self, d):
		self._shapes = torch.zeros(d)
		self._rates = torch.zeros(d)

		self._initialized = True
		super()._initialize(d)

	def _reset_cache(self):
		if self._initialized == False:
			return

		self._w_sum = torch.zeros(self.d)
		self._xw_sum = torch.zeros(self.d)
		self._logx_w_sum = torch.zeros(self.d)

		self._log_rates = torch.log(self.rates)
		self._lgamma_shapes = torch.lgamma(self.shapes)
		self._thetas = self._log_rates * self.shapes - self._lgamma_shapes

	def log_probability(self, X):
		X = _cast_as_tensor(X)
		return torch.sum(self._thetas + torch.log(X) * (self.shapes - 1) - 
			self.rates * X, dim=-1)

	def summarize(self, X, sample_weights=None):
		X, sample_weights = super().summarize(X, sample_weights=sample_weights)

		self._w_sum += torch.sum(sample_weights, axis=(0, 1))
		self._xw_sum += torch.sum(X * sample_weights, dim=0)
		self._logx_w_sum += torch.sum(torch.log(X) * sample_weights, dim=0)

	def from_summaries(self):
		thetas = torch.log(self._xw_sum / self._w_sum) - \
			self._logx_w_sum / self._w_sum

		numerator = (3 - thetas + torch.sqrt((thetas - 3) ** 2 + 24 * thetas))
		denominator = (12 * thetas)

		new_shapes = numerator / denominator
		shapes = new_shapes + self.tol

		for iteration in range(self.max_iter):
			mask = torch.abs(shapes - new_shapes) < self.tol
			if torch.all(mask):
				break

			shapes = new_shapes
			new_shapes = (shapes - (torch.log(shapes) - torch.polygamma(0, 
				shapes) - thetas) / (1.0 / shapes - torch.polygamma(1, shapes)))

		shapes = new_shapes
		rates = 1.0 / (1.0 / (shapes * self._w_sum) * self._xw_sum)

		_update_parameter(self._shapes, torch.log(shapes), self.inertia)
		_update_parameter(self._rates, torch.log(rates), self.inertia)
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

from pomegranate import IndependentComponentsDistribution
from pomegranate import GammaDistribution

d = 4500
n = 10000

mu = numpy.random.randn(d) * 15
X = numpy.exp(numpy.random.randn(n, d))

tic = time.time()
d1 = IndependentComponentsDistribution.from_samples(X, distributions=GammaDistribution)
logp1 = d1.log_probability(X)
toc1 = time.time() - tic

muz = torch.tensor([d.parameters[0] for d in d1.distributions])
X = torch.tensor(X, dtype=torch.float32)

print(muz)

tic = time.time()
d2 = Gamma()
d2.summarize(X)
d2.from_summaries()
logp2 = d2.log_probability(X)
toc2 = time.time() - tic

print("Gamma Distribution Fitting and Logp")
print("pomegranate time: {:4.4}, pomegranate logp: {:4.4}".format(toc1, logp1.sum()))
print("torchegranate time: {:4.4}, torchegranate logp: {:4.4}".format(toc2, logp2.sum()))



n = 10000
d = 12

X = torch.randn(n, d)
X = torch.exp(X)

model = ToyNet(d)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for i in range(10):
	optimizer.zero_grad()
	
	shapes, rates = model(X)
	loss = -Gamma(shapes, 2).log_probability(X).sum()

	print(loss.item())
	loss.backward()
	optimizer.step()
