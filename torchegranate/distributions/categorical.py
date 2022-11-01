# categorical.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import torch

from .._utils import _cast_as_tensor
from .._utils import _update_parameter
from .._utils import _check_parameter
from .._utils import _reshape_weights

from ._distribution import Distribution


class Categorical(Distribution):
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

	def __init__(self, probs=None, inertia=0.0, frozen=False):
		super().__init__(inertia=inertia, frozen=frozen)
		self.name = "Categorical"

		self.probs = _check_parameter(_cast_as_tensor(probs), "probs", 
			min_value=0, max_value=1, ndim=2)

		self._initialized = probs is not None
		self.d = self.probs.shape[0] if self._initialized else None
		self.n_keys = self.probs.shape[1] if self._initialized else None
		self._reset_cache()

	def _initialize(self, d, n_keys):
		self.probs = torch.zeros(d, n_keys)

		self.n_keys = n_keys
		self._initialized = True
		super()._initialize(d)

	def _reset_cache(self):
		if self._initialized == False:
			return

		self._w_sum = torch.zeros(self.d)
		self._xw_sum = torch.zeros(self.d, self.n_keys)

		self._log_probs = torch.log(self.probs)

	def log_probability(self, X):
		X = _check_parameter(_cast_as_tensor(X), "X", min_value=0.0,
			max_value=self.n_keys-1, ndim=2, shape=(-1, self.d))

		logps = torch.zeros(X.shape[0], dtype=self.probs.dtype)
		for i in range(self.d):
			logps += self._log_probs[i][X[:, i]]

		return logps

	def _summarize(self, X, sample_weight):
		X = _cast_as_tensor(X)
		sample_weight = _reshape_weights(X, _cast_as_tensor(sample_weight))
		
		if not self._initialized:
			self._initialize(len(X[0]), int(X.max())+1)

		_check_parameter(sample_weight, "sample_weight", min_value=0, 
			shape=(-1, self.d))

		return X, sample_weight

	def summarize(self, X, sample_weight=None):
		if self.frozen == True:
			return

		X, sample_weight = self._summarize(X, sample_weight=sample_weight)
		X = _check_parameter(X, "X", min_value=0, max_value=self.n_keys-1)

		self._w_sum += torch.sum(sample_weight, dim=0)
		for i in range(self.n_keys):
			self._xw_sum[:, i] += torch.sum((X == i) * sample_weight, dim=0)

	def from_summaries(self):
		if self.frozen == True:
			return

		probs = self._xw_sum / self._w_sum.unsqueeze(1)

		_update_parameter(self.probs, probs, self.inertia)
		self._reset_cache()
