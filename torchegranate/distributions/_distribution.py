# _distribution.py
# Jacob Schreiber <jmschreiber91@gmail.com>

import torch

from ._utils import _cast_as_tensor
from ._utils import _update_parameter
from ._utils import _check_parameter


class Distribution(torch.nn.Module):
	"""A base distribution object.

	This distribution is inherited by all the other distributions.
	"""

	def __init__(self, inertia, frozen):
		super(Distribution, self).__init__()

		self.inertia = _check_parameter(inertia, "inertia", min_value=0, 
			max_value=1, ndim=0)
		self.frozen = _check_parameter(frozen, "frozen", 
			value_set=[True, False], ndim=0) 

		self._initialized = False

	def forward(self, X):
		self.summarize(X)
		return self.log_probability(X)

	def backward(self, X):
		self.from_summaries()
		return X

	def _initialize(self, d):
		self.d = d
		self._reset_cache()

	def _reset_cache(self):
		raise NotImplementedError

	def probability(self, X):
		return torch.exp(self.log_probability(X))

	def log_probability(self, X):
		raise NotImplementedError

	def fit(self, X, sample_weights=None):
		self.summarize(X, sample_weights=sample_weights)
		self.from_summaries()
		return self

	def summarize(self, X, sample_weights=None):
		if not self._initialized:
			self._initialize(len(X[0]))

		X = _cast_as_tensor(X)
		sample_weights = _cast_as_tensor(sample_weights)

		if sample_weights is None:
			sample_weights = torch.ones(*X.shape)
		elif len(sample_weights.shape) == 1: 
			sample_weights = sample_weights.reshape(-1, 1).expand(-1, 
				X.shape[1])
		elif sample_weights.shape[1] == 1:
			sample_weights = sample_weights.expand(-1, X.shape[1])
		elif sample_weights.shape[1] != X.shape[-1]:
			raise ValueError("Variable sample_weight must have shape equal"
				" to X or have the second dimension be 1.")

		if X.shape[0] != sample_weights.shape[0]:
			raise ValueError("Variables X and sample_weight must have an "
				"equal number of elements in the first dimension.")

		sample_weights = _check_parameter(sample_weights, "sample_weights", 
			min_value=0, shape=(-1, self.d))

		return X, sample_weights

	def from_summaries(self):
		raise NotImplementedError
