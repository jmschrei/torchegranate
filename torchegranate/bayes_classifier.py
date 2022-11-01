# BayesClassifier.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

import torch

from ._utils import _cast_as_tensor
from ._utils import _update_parameter
from ._utils import _check_parameter
from ._utils import _reshape_weights

from ._bayes import BayesMixin

from .distributions._distribution import Distribution


class BayesClassifier(BayesMixin, Distribution):
	def __init__(self, distributions, priors=None, inertia=0.0, frozen=False):
		super().__init__(inertia=inertia, frozen=frozen)
		self.name = "BayesClassifier"

		self.distributions = _check_parameter(distributions, "distributions",
			dtypes=(list, tuple))

		self.priors = _check_parameter(_cast_as_tensor(priors), "priors", 
			min_value=0, max_value=1, ndim=1, value_sum=1.0, 
			shape=(len(distributions),))

		self.m = len(distributions)

		if all(d._initialized for d in distributions):
			self._initialized = True
			self.d = distributions[0].d
			if self.priors is None:
				self.priors = torch.ones(self.m) / self.m

		else:
			self._initialized = False
			self.d = None
		
		self._reset_cache()

	def summarize(self, X, y, sample_weight=None):
		X, sample_weight = super().summarize(X, sample_weight=sample_weight)
		y = _check_parameter(_cast_as_tensor(y), "y", min_value=0, 
			max_value=self.m-1, ndim=1, shape=(len(X),))
		sample_weight = _check_parameter(sample_weight, "sample_weight", 
			min_value=0, shape=(-1, self.d))

		for j, d in enumerate(self.distributions):
			idx = y == j
			d.summarize(X[idx], sample_weight[idx])

			if self.frozen == False:
				self._w_sum[j] += sample_weight[idx].mean(dim=-1).sum()
