# _distribution.py
# Jacob Schreiber <jmschreiber91@gmail.com>

import torch

from _utils import _cast_as_tensor

class Distribution(torch.nn.Module):
	"""A base distribution object.

	This distribution is inherited by all the other distributions.
	"""

	def __init__(self):
		super(Distribution, self).__init__()
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

	def log_probability(self, X):
		raise NotImplementedError

	def fit(self, X, sample_weights=None):
		self.summarize(X, sample_weights=sample_weights)
		self.from_summarize()

	def summarize(self, X, sample_weights=None):
		if not self._initialized:
			self._initialize(X.shape[1])

		X = _cast_as_tensor(X)
		sample_weights = _cast_as_tensor(sample_weights)

		if sample_weights is None:
			sample_weights = torch.ones(X.shape[0], 1)

		return X, sample_weights

	def from_summaries(self):
		raise NotImplementedError
