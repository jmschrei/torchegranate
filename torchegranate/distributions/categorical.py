# multinomial.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import torch

class Multinomial():
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
			self.n_keys, self.d = None, None
			self._initialized = False
		else:
			self.log_probabilities = torch.log(probabilities)
			self.n_keys, self.d = probabilities.shape
			self._initialized = True

		self.frozen = frozen
		self.inertia = inertia
		self._reset_cache()

	def _initialize(self, k, d):
		self.log_probabilities = torch.log(torch.ones((k, d)) / k)
		self.n_keys = k
		self.d = d
		self._reset_cache()

	def _reset_cache(self):
		self._counts = torch.zeros_like(self.log_probabilities)

	def log_probability(self, X):
		logps = torch.zeros(X.shape[0])
		for i in range(self.d):
			logps += self.log_probabilities[:, i][X[:, i]]
		return logps

	def summarize(self, X, sample_weights=None):
		for i in range(self.n_keys):
			self._counts[i] += (X == i).sum(axis=0)

	def from_summaries(self):
		self.log_probabilities = torch.log(self._counts / self._counts.sum(axis=0, keepdims=True))

'''
import numpy
import time 

from pomegranate import MultivariateGaussianDistribution
from pomegranate import IndependentComponentsDistribution
from pomegranate import DiscreteDistribution

d = 50
n = 100000
k = 20

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
#d2.summarize(X)
#d2.from_summaries()
logp2 = d2.log_probability(X)
toc2 = time.time() - tic

print(toc1, logp1.sum())
print(toc2, logp2.sum())
print(numpy.abs(logp1 - logp2.numpy()).sum())
'''