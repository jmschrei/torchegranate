# gmm.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

import torch

from ._utils import _cast_as_tensor
from ._utils import _update_parameter
from ._utils import _check_parameter

from .distributions._distribution import Distribution

from ._bayes import BayesMixin


class GeneralMixtureModel(BayesMixin, Distribution):
	def __init__(self, distributions, priors=None, max_iter=10, threshold=0.1, 
		inertia=0.0, frozen=False):
		super().__init__(inertia=inertia, frozen=frozen)
		self.name = "GeneralMixtureModel"

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
		
		self.max_iter = max_iter
		self.threshold = threshold
		self._reset_cache()

	def fit(self, X, sample_weight=None):
		initial_logp = self.log_probability(X).sum()
		logp = initial_logp

		for i in range(self.max_iter):
			self.summarize(X, sample_weight=sample_weight)
			self.from_summaries()			

			last_logp = logp
			logp = self.log_probability(X).sum()

			if logp - last_logp < self.threshold:
				break

		return self

	def summarize(self, X, sample_weight=None):
		X, sample_weight = super().summarize(X, sample_weight=sample_weight)
		sample_weight = _check_parameter(sample_weight, "sample_weight", 
			min_value=0, shape=(-1, self.d))

		e = self.predict_proba(X)
		for i, d in enumerate(self.distributions):
			d.summarize(X, e[:, i:i+1] * sample_weight)

			if self.frozen == False:
				self._w_sum[i] += (e[:, i:i+1] * 
					sample_weight).mean(dim=-1).sum()
