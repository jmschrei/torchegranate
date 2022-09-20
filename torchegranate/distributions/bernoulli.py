# bernoulli.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import torch

class Bernoulli():
	"""A multinomial distribution object.

	This distribution represents a categorical distribution over features. The
	keys must be contiguous non-negative integers that begin at zero. Because
	the probabilities are represented as a single tensor, there must be one
	probability value in each feature up to the maximum key in any feature, but
	those probabilities can be zero. 

	Parameters
	----------
	probabilities: torch.tensor or None, shape=(k, d), optional
		Probabilities for each key for each feature, where k is the largest
		number of keys across all features. 

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

	def __init__(self, probabilities=None, frozen=False, inertia=0.0):
		if probabilities is None:
			self.log_probabilities = None
			self.d = None
			self._initialized = False
		else:
			self.log_probabilities = torch.log(probabilities)
			self.d = probabilities.shape[0]
			self._initialized = True

		self.frozen = frozen
		self.inertia = inertia
		self._reset_cache()

	def _initialize(self, d):
		self.log_probabilities = torch.log(torch.ones(d) / k)
		self.d = d
		self._reset_cache()

	def _reset_cache(self):
		self._inv_log_probabilities = torch.log(1. - torch.exp(self.log_probabilities))
		self._counts = torch.zeros_like(self.log_probabilities)
		self._total_counts = 0

	def log_probability(self, X):
		return X.matmul(self.log_probabilities) + (1-X).matmul(self._inv_log_probabilities)

	def summarize(self, X, sample_weights=None):
		self._total_counts += X.shape[0]
		self._counts += torch.sum(X, dim=0)

	def from_summaries(self):
		self.log_probabilities = torch.log(self._counts / self._total_counts)
		self._inv_log_probabilities = torch.log(1. - self._counts / self._total_counts)
		self._reset_cache()


import numpy
import time 

from pomegranate import MultivariateGaussianDistribution
from pomegranate import IndependentComponentsDistribution
from pomegranate import DiscreteDistribution

from categorical import Multinomial

d = 100
n = 50000
k = 2

X = numpy.random.choice(k, size=(n, d))
mu = numpy.abs(numpy.random.randn(k, d))
mu = mu / mu.sum(axis=0, keepdims=True)

tic = time.time()
d1 = IndependentComponentsDistribution.from_samples(X, distributions=DiscreteDistribution)
logp1 = d1.log_probability(X)
toc1 = time.time() - tic

X = torch.tensor(X, dtype=torch.int64)
mu = torch.tensor(mu, dtype=torch.float32)


tic = time.time()
d2 = Multinomial(mu)
d2.summarize(X)
d2.from_summaries()
logp2 = d2.log_probability(X)
toc2 = time.time() - tic


X = torch.tensor(X, dtype=torch.float32)

tic = time.time()
d3 = Bernoulli(mu[1])
d3.summarize(X)
d3.from_summaries()
logp3 = d3.log_probability(X)
toc3 = time.time() - tic


print("Categorical Distribution Fitting and Logp")
print("pomegranate time: {:4.4}, pomegranate logp: {:4.4}".format(toc1, logp1.sum()))
print("torchegranate (categorical) time: {:4.4}, torchegranate logp: {:4.4}".format(toc2, logp2.sum()))
print("torchegranate (bernoulli) time: {:4.4}, torchegranate logp: {:4.4}".format(toc3, logp3.sum()))

print(numpy.abs(logp1 - logp2.numpy()).sum())
