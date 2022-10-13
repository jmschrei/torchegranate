# poisson.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import torch

from _utils import _cast_as_tensor
from _utils import _update_parameter

from _distribution import Distribution

class Poisson(Distribution):
	"""An poisson distribution object.

	A poisson distribution models the number of occurances of events that
	happen in a fixed time span, assuming that the occurance of each event
	is independent. This distibution also asumes that each feature is
	independent of the others.

	There are two ways to initialize this objecct. The first is to pass in
	the tensor of lambda parameters, at which point they can immediately be
	used. The second is to not pass in the lambda parameters and then call
	either `fit` or `summary` + `from_summaries`, at which point the lambda
	parameter will be learned from data.


	Parameters
	----------
	lambdas: torch.tensor or None, shape=(d,), optional
		The lambda parameters for each feature. Default is None.

	inertia: float, (0, 1), optional
		Indicates the proportion of the update to apply to the parameters
		during training. When the inertia is 0.0, the update is applied in
		its entirety and the previous parameters are ignored. When the
		inertia is 1.0, the update is entirely ignored and the previous
		parameters are kept, equivalently to if the parameters were frozen.

	frozen: bool, optional
		Whether all the parameters associated with this distribution are frozen.
		If you want to freeze individual pameters, or individual values in those
		parameters, you must modify the `frozen` attribute of the tensor or
		parameter directly. Default is False.


	Examples
	--------
	>>> # Create a distribution with known parameters
	>>> rates = torch.tensor([1.2, 0.4])
	>>> X = torch.tensor([[0.3, 0.2], [0.8, 0.1]])
	>>>
	>>> d = Gamma(rates)
	>>> d.log_probability(X)
	tensor([-1.1740, -1.7340])
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


	def __init__(self, lambdas, inertia=0.0, frozen=False):
		super().__init__()
		self.name = "Poisson"
		self.inertia = inertia
		self.frozen = frozen

		self.lambdas = _cast_as_tensor(lambdas)

		self._initialized = lambdas is not None
		self.d = len(self.lambdas) if self._initialized else None
		self._reset_cache()

	def _initialize(self, d):
		self.lambdas = torch.zeros(d)

		self._initialized = True
		super()._initialize(d)

	def _reset_cache(self):
		if self._initialized == False:
			return

		self._w_sum = torch.zeros(self.d)
		self._xw_sum = torch.zeros(self.d)

		self._log_lambdas = torch.log(self.lambdas)

	def log_probability(self, X):
		X = _cast_as_tensor(X)
		return torch.sum(X * self._log_lambdas - self.lambdas - 
			torch.lgamma(X+1), dim=-1)

	def summarize(self, X, sample_weights=None):
		if self.frozen == True:
			return

		X, sample_weights = super().summarize(X, sample_weights=sample_weights)

		self._w_sum += torch.sum(sample_weights, axis=(0, 1))
		self._xw_sum += torch.sum(X * sample_weights, dim=0)

	def from_summaries(self):
		if self.frozen == True:
			return

		lambdas = self._xw_sum / self._w_sum
		_update_parameter(self.lambdas, lambdas, self.inertia)
		self._reset_cache()


import numpy
import time 

from pomegranate import IndependentComponentsDistribution
from pomegranate import PoissonDistribution

d = 450
n = 1000

mu = numpy.random.randn(d) * 15
X = numpy.exp(numpy.random.randn(n, d))

tic = time.time()
d1 = IndependentComponentsDistribution.from_samples(X, distributions=PoissonDistribution)
logp1 = d1.log_probability(X)
toc1 = time.time() - tic

muz = torch.tensor([d.parameters[0] for d in d1.distributions])
X = torch.tensor(X, dtype=torch.float32)

tic = time.time()
d2 = Poisson(muz)
d2.summarize(X)
d2.from_summaries()
logp2 = d2.log_probability(X)
toc2 = time.time() - tic

print(toc1, logp1.sum())
print(toc2, logp2.sum())
print(numpy.abs(logp1 - logp2.numpy()).sum())

torch.save(d2, "test.torch")

d3 = torch.load("test.torch")

tic = time.time()
logp3 = d3.log_probability(X)
toc3 = time.time() - tic

print(toc3, logp3.sum())
print(numpy.abs(logp1 - logp3.numpy()).sum())