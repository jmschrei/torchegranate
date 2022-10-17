# diracdelta.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>


import torch

from ._utils import _cast_as_tensor
from ._utils import _update_parameter

from ._distribution import Distribution


class DiracDelta(Distribution):
	def __init__(self, alpha=0.0, inertia=0.0, frozen=False):
		super().__init__()
		self.alpha = alpha

	def _initialize(self, d):
		self.alpha = 0.0

		self._initialized = True
		super()._initialize(d)

	def _reset_cache(self):
		return

	def log_probability(self, X):
		X = _cast_as_tensor(X)
		return torch.sum(torch.where(X == 0.0, self.alpha, float("-inf")), dim=-1)

	def summarize(self, X, sample_weights=None):
		return

	def from_summaries(self):
		return
