# multinomial.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>


import torch

from _utils import _cast_as_tensor
from _utils import _update_parameter

from _distribution import Distribution


class Multinomial(Distribution):
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

	"""

	def __init__(self, probabilities=None, inertia=0.0, frozen=False):
		super().__init__()
		self.name = "Categorical"
		self.inertia = inertia
		self.frozen = frozen

		self.probabilities = _cast_as_tensor(probabilities)

		self._initialized = probabilities is not None
		self.d = self.probabilities.shape[1] if self._initialized else None
		self.n_keys = self.probabilities.shape[0] if self._initialized else None
		self._reset_cache()

	def _initialize(self, k, d):
		self.probabilities = torch.zeros(d)

		self.n_keys = k
		self._initialized = True
		super()._initialize(d)


	def _reset_cache(self):
		if self._initialized == False:
			return

		self._log_probabilities = torch.log(self.probabilities)
		self._counts = torch.zeros(self.n_keys, d)

	def log_probability(self, X):
		X = _cast_as_tensor(X)

		logps = torch.zeros(X.shape[0])
		for i in range(self.d):
			logps += self._log_probabilities[:, i][X[:, i]]

		return logps

	def summarize(self, X, sample_weights=None):
		if self.frozen == True:
			return

		X, sample_weights = super().summarize(X, sample_weights=sample_weights)

		for i in range(self.n_keys):
			self._counts[i] += (X == i).sum(axis=0)

	def from_summaries(self):
		if self.frozen == True:
			return

		probabilities = (self._counts / 
			self._counts.sum(axis=0, keepdims=True))

		_update_parameter(self.probabilities, probabilities, self.inertia)
		self._reset_cache()


import numpy
import time 

from pomegranate import MultivariateGaussianDistribution
from pomegranate import IndependentComponentsDistribution
from pomegranate import DiscreteDistribution

d = 50
n = 1000
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
d2.summarize(X)
d2.from_summaries()
logp2 = d2.log_probability(X)
toc2 = time.time() - tic



print("Categorical Distribution Fitting and Logp")
print("pomegranate time: {:4.4}, pomegranate logp: {:4.4}".format(toc1, logp1.sum()))
print("torchegranate time: {:4.4}, torchegranate logp: {:4.4}".format(toc2, logp2.sum()))
print(numpy.abs(logp1 - logp2.numpy()).sum())
