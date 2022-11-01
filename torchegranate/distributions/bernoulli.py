# bernoulli.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import torch

from .._utils import _cast_as_tensor
from .._utils import _update_parameter
from .._utils import _check_parameter
from .._utils import eps

from ._distribution import Distribution


class Bernoulli(Distribution):
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
	probs: torch.tensor or None, shape=(d,), optional
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

	def __init__(self, probs=None, inertia=0.0, frozen=False):
		super().__init__(inertia=inertia, frozen=frozen)
		self.name = "Bernoulli"

		self.probs = _check_parameter(_cast_as_tensor(probs), "probs", 
			min_value=eps, max_value=1-eps, ndim=1)

		self._initialized = self.probs is not None
		self.d = len(self.probs) if self._initialized else None
		self._reset_cache()

	def _initialize(self, d):
		self.probs = torch.zeros(d)

		self._initialized = True
		super()._initialize(d)

	def _reset_cache(self):
		if self._initialized == False:
			return

		self._w_sum = torch.zeros(self.d)
		self._xw_sum = torch.zeros(self.d)

		self._log_probs = torch.log(self.probs)
		self._log_inv_probs = torch.log(1-self.probs)

	def log_probability(self, X):
		X = _check_parameter(_cast_as_tensor(X, dtype=self.probs.dtype), "X", 
			value_set=(0, 1), ndim=2, shape=(-1, self.d))

		return X.matmul(self._log_probs) + (1-X).matmul(self._log_inv_probs)

	def summarize(self, X, sample_weight=None):
		if self.frozen == True:
			return

		X, sample_weight = super().summarize(X, sample_weight=sample_weight)
		X = _check_parameter(X, "X", value_set=(0, 1))

		self._w_sum += torch.sum(sample_weight, dim=0)
		self._xw_sum += torch.sum(X * sample_weight, dim=0)

	def from_summaries(self):
		if self.frozen == True:
			return

		probs = self._xw_sum / self._w_sum
		_update_parameter(self.probs, probs, self.inertia)
		self._reset_cache()
