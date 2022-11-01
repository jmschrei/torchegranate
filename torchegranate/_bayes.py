# _bayes.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>


import torch

from ._utils import _cast_as_tensor
from ._utils import _update_parameter
from ._utils import _check_parameter
from ._utils import _reshape_weights

from .distributions._distribution import Distribution


class BayesMixin(torch.nn.Module):
	def _initialize(self, d):
		self.priors = torch.ones(self.m) / self.m

		self._initialized = True
		super()._initialize(d)

	def _reset_cache(self):
		if self._initialized == False:
			return

		self._w_sum = torch.zeros(self.m)
		self._log_priors = torch.log(self.priors)

	def _emission_matrix(self, X):
		X = _check_parameter(_cast_as_tensor(X), "X", ndim=2, 
			shape=(-1, self.d))

		e = torch.empty(X.shape[0], self.m)
		for i, d in enumerate(self.distributions):
			e[:, i] = d.log_probability(X)

		return e + self._log_priors

	def log_probability(self, X):
		e = self._emission_matrix(X)
		return torch.logsumexp(e, dim=1)

	def predict(self, X):
		e = self._emission_matrix(X)
		return torch.argmax(e, dim=1)

	def predict_proba(self, X):
		e = self._emission_matrix(X)
		return torch.exp(e - torch.logsumexp(e, dim=1, keepdims=True))
		
	def predict_log_proba(self, X):
		e = self._emission_matrix(X)
		return e - torch.logsumexp(e, dim=1, keepdims=True)

	def fit(self, X, y, sample_weight=None):
		self.summarize(X, y, sample_weight=sample_weight)
		self.from_summaries()
		return self

	def from_summaries(self):
		for d in self.distributions:
			d.from_summaries()

		if self.frozen == True:
			return

		priors = self._w_sum / torch.sum(self._w_sum)

		_update_parameter(self.priors, priors, self.inertia)
		self._reset_cache()
