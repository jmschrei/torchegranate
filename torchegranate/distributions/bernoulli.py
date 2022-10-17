# bernoulli.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import torch

class Bernoulli():
	"""A Bernoulli distribution object.

	A Bernoulli distribution models the probability of a binary variable
	occurring. rates of discrete events, and has a probability parameter
	describing this value. This distribution assumes that each feature is 
	independent of the others.

	There are two ways to initialize this object. The first is to pass in
	the tensor of probablity parameters, at which point they can immediately be
	used. The second is to not pass in the rate parameters and then call
	either `fit` or `summary` + `from_summaries`, at which point the probability
	parameter will be learned from data.


	Parameters
	----------
	probabilities: torch.tensor or None, shape=(d,), optional
		The rate parameters for each feature. Default is None.

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
	"""

	def __init__(self, probabilities=None, inertia=0.0, frozen=False):
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
