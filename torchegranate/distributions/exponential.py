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

		self._weight_sum = torch.zeros(self.d)
		self._column_w_sum = torch.zeros(self.d)

	def __reduce__(self):
		"""Serialize distribution for pickling."""
		return self.__class__, (self.rates, self.frozen, self.min_std)

	def log_probability(self, X):
		return torch.sum(self._log_rates - self.rates * X, dim=1)

	def summarize(self, X, sample_weights=None):
		self._weight_sum = torch.sum(sample_weights)
		self._column_w_sum = torch.sum(X.T * sample_weights, dim=1)

	def from_summaries(self):
		if self._weight_sum < 1e-8 or self.frozen == True:
			return

		self.rates = self._column_w_sum / self._weight_sum
		self._log_rates = torch.log(self.rates)



import numpy
import time 

from pomegranate import MultivariateGaussianDistribution
from pomegranate import IndependentComponentsDistribution
from pomegranate import NormalDistribution

d = 1500
n = 100000

z = 3.7784

mu = numpy.random.randn(d) * 15
cov = numpy.eye(d) * z #* numpy.abs(numpy.random.randn(d))


X = numpy.random.randn(n, d)

tic = time.time()
d1 = IndependentComponentsDistribution.from_samples(X, distributions=NormalDistribution)
logp1 = d1.log_probability(X)
toc1 = time.time() - tic

X = torch.tensor(X, dtype=torch.float32)

tic = time.time()
d2 = Normal(mu, z, covariance_type='diag')
d2.summarize(X[:100])
d2.summarize(X[100:5000])
d2.summarize(X[5000:])
d2.from_summaries()
logp2 = d2.log_probability(X)
toc2 = time.time() - tic

print(toc1, logp1.sum())
print(toc2, logp2.sum())
print(numpy.abs(logp1 - logp2.numpy()).sum())