# exponential.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import torch

# Define some useful constants
NEGINF = float("-inf")
INF = float("inf")

class Exponential():
	"""A normal distribution based on a mean and standard deviation."""
	def __init__(self, rates, frozen=False, inertia=0.0):
		self.d = len(rates)
		self.rates = torch.tensor(rates)
		self._log_rates = torch.log(self.rates)
		self.name = "Exponential"
		self.frozen = frozen
		self.inertia = inertia

		self._w_sum = torch.zeros(self.d)
		self._xw_sum = torch.zeros(self.d)

	def __reduce__(self):
		return self.__class__, (self.rates, self.frozen)

	def log_probability(self, X):
		return torch.sum(self._log_rates - self.rates * X, dim=1)

	def summarize(self, X, sample_weights=None):
		if sample_weights is None:
			sample_weights = torch.ones(X.shape[0], 1)

		self._w_sum += torch.sum(sample_weights, axis=(0, 1))
		self._xw_sum += torch.sum(X * sample_weights, dim=0)

	def from_summaries(self):
		if self._w_sum.sum() < 1e-8 or self.frozen == True:
			return

		rates = self._w_sum / self._xw_sum

		self.rates = self._w_sum / self._xw_sum
		self._log_rates = torch.log(self.rates)

		self._w_sum = torch.zeros(self.d)
		self._xw_sum = torch.zeros(self.d)

		#self.clear_summaries()


import numpy
import time 

from pomegranate import MultivariateGaussianDistribution
from pomegranate import IndependentComponentsDistribution
from pomegranate import ExponentialDistribution

d = 1500
n = 10000

mu = numpy.random.randn(d) * 15
X = numpy.exp(numpy.random.randn(n, d))

tic = time.time()
d1 = IndependentComponentsDistribution.from_samples(X, distributions=ExponentialDistribution)
logp1 = d1.log_probability(X)
toc1 = time.time() - tic

muz = torch.tensor([d.parameters[0] for d in d1.distributions])
X = torch.tensor(X, dtype=torch.float32)

tic = time.time()
d2 = Exponential(mu)
d2.summarize(X)
d2.from_summaries()
logp2 = d2.log_probability(X)
toc2 = time.time() - tic

print("Exponential Distribution Fitting and Logp")
print("pomegranate time: {:4.4}, pomegranate logp: {:4.4}".format(toc1, logp1.sum()))
print("torchegranate time: {:4.4}, torchegranate logp: {:4.4}".format(toc2, logp2.sum()))

print(numpy.abs(logp1 - logp2.numpy()).sum())
