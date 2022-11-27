# gmm.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

import time
import torch

from ._utils import _cast_as_tensor
from ._utils import _update_parameter
from ._utils import _check_parameter
from ._utils import _reshape_weights

from .distributions._distribution import Distribution

from ._bayes import BayesMixin

from .kmeans import KMeans


class GeneralMixtureModel(BayesMixin, Distribution):
	def __init__(self, distributions, priors=None, init='random', max_iter=10, 
		tol=0.1, inertia=0.0, frozen=False, random_state=None, verbose=False):
		super().__init__(inertia=inertia, frozen=frozen)
		self.name = "GeneralMixtureModel"

		self.distributions = _check_parameter(distributions, "distributions",
			dtypes=(list, tuple))

		self.priors = _check_parameter(_cast_as_tensor(priors), "priors", 
			min_value=0, max_value=1, ndim=1, value_sum=1.0, 
			shape=(len(distributions),))

		self.verbose = verbose

		self.k = len(distributions)

		if all(d._initialized for d in distributions):
			self._initialized = True
			self.d = distributions[0].d
			if self.priors is None:
				self.priors = torch.ones(self.k) / self.k

		else:
			self._initialized = False
			self.d = None
		
		self.init = init
		self.max_iter = max_iter
		self.tol = tol
		self.random_state = random_state
		self._reset_cache()

	def _initialize(self, X):
		X = _check_parameter(_cast_as_tensor(X), "X", ndim=2)
		y_hat = KMeans(self.k, init=self.init, max_iter=3, 
			random_state=self.random_state).fit_predict(X)

		self.priors = torch.empty(self.k)

		for i in range(self.k):
			idx = y_hat == i

			self.distributions[i].fit(X[idx])
			self.priors[i] = idx.type(torch.float32).mean()

		self._initialized = True
		self._reset_cache()
		super()._initialize(X.shape[1])

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
		logp = torch.logsumexp(e, dim=1, keepdims=True)
		y = torch.exp(e - logp)

		for i, d in enumerate(self.distributions):
			d.summarize(X, y[:, i:i+1] * sample_weight)

			if self.frozen == False:
				self._w_sum[i] += (y[:, i:i+1] * 
					sample_weight).mean(dim=-1).sum()

		return torch.sum(logp)
