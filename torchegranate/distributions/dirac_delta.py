# diracdelta.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import torch

from .._utils import _cast_as_tensor
from .._utils import _update_parameter
from .._utils import _check_parameter

from ._distribution import Distribution


class DiracDelta(Distribution):
	def __init__(self, alphas=None, inertia=0.0, frozen=False):
		super().__init__(inertia=inertia, frozen=frozen)
		self.name = "DiracDelta"

		self.alphas = _check_parameter(_cast_as_tensor(alphas), "alphas", 
			min_value=0.0, ndim=1)

		self._initialized = alphas is not None
		self.d = len(self.alphas) if self._initialized else None
		self._reset_cache()

	def _initialize(self, d):
		self.alphas = torch.ones(d)

		self._initialized = True
		super()._initialize(d)

	def _reset_cache(self):
		if self._initialized == False:
			return

		self._log_alphas = torch.log(self.alphas)

	def log_probability(self, X):
		X = _check_parameter(_cast_as_tensor(X), "X", ndim=2, 
			shape=(-1, self.d))

		return torch.sum(torch.where(X == 0.0, self._log_alphas, float("-inf")), 
			dim=-1)

	def summarize(self, X, sample_weight=None):
		if self.frozen == True:
			return

		X, sample_weight = super().summarize(X, sample_weight=sample_weight)

	def from_summaries(self):
		return
