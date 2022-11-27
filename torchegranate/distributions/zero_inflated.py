# zeroinflated.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import time
import torch

from .._utils import _cast_as_tensor
from .._utils import _update_parameter
from .._utils import _check_parameter
from .._utils import _reshape_weights

from ._distribution import Distribution


class ZeroInflated(Distribution):
	def __init__(self, distribution, priors=None, max_iter=10, 
		tol=0.1, inertia=0.0, frozen=False, verbose=False):
		super().__init__(inertia=inertia, frozen=frozen)
		self.name = "ZeroInflated"

		self.distribution = distribution
		self.priors = _check_parameter(_cast_as_tensor(priors), "priors", 
			min_value=0, max_value=1, ndim=1, value_sum=1.0)

		self.verbose = verbose
		self._initialized = distribution._initialized is True
		self.d = distribution.d if self._initialized else None
		
		self.max_iter = max_iter
		self.tol = tol
		self._reset_cache()

	def _initialize(self, X):
		self.distribution._initialize(X.shape[1])
		self.distribution.fit(X)

		self.priors = torch.ones(X.shape[1]) / 2
		self._initialized = True
		super()._initialize(X.shape[1])

	def _reset_cache(self):
		if self._initialized == False:
			return

		self._w_sum = torch.zeros(self.d, 2)
		self._log_priors = torch.log(self.priors)

	def _emission_matrix(self, X):
		X = _check_parameter(_cast_as_tensor(X), "X", ndim=2, 
			shape=(-1, self.d))

		e = torch.empty(X.shape[0], self.d, 2)
		e[:, :, 0] = self._log_priors.unsqueeze(0)
		e[:, :, 0] += self.distribution.log_probability(X).unsqueeze(1)
		
		e[:, :, 1] = torch.log(1 - self.priors).unsqueeze(0)
		e[:, :, 1] += torch.where(X == 0, 0, float("-inf"))
		return e

	def fit(self, X, sample_weight=None):
		logp = None
		for i in range(self.max_iter):
			start_time = time.time()

			last_logp = logp
			logp = self.summarize(X, sample_weight=sample_weight)

			if i > 0:
				improvement = logp - last_logp
				duration = time.time() - start_time

				if self.verbose:
					print("[{}] Improvement: {}, Time: {:4.4}s".format(i, 
						improvement, duration))

				if improvement < self.tol:
					break

			self.from_summaries()

		self._reset_cache()
		return self

	def summarize(self, X, sample_weight=None):
		X = _cast_as_tensor(X)
		if not self._initialized:
			self._initialize(X)

		_check_parameter(X, "X", ndim=2, shape=(-1, self.d))
		sample_weight = _reshape_weights(X, _cast_as_tensor(sample_weight, 
			dtype=torch.float32))

		e = self._emission_matrix(X)
		logp = torch.logsumexp(e, dim=2, keepdims=True)
		y = torch.exp(e - logp)

		self.distribution.summarize(X, y[:, :, 0] * sample_weight)

		if not self.frozen:
			self._w_sum += torch.sum(y * sample_weight.unsqueeze(-1), dim=(0, 1)) 

		return torch.sum(logp)

	def from_summaries(self):
		self.distribution.from_summaries()

		if self.frozen == True:
			return

		priors = self._w_sum[:,0] / torch.sum(self._w_sum, dim=-1)

		_update_parameter(self.priors, priors, self.inertia)
		self._reset_cache()
