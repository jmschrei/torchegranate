# kmeans.py
# Author: Jacob Schreiber

import time
import torch

from ._utils import _cast_as_tensor
from ._utils import _update_parameter
from ._utils import _check_parameter
from ._utils import _reshape_weights
from ._utils import _initialize_centroids

from ._utils import eps

class KMeans(torch.nn.Module):
	def __init__(self, k=None, centroids=None, init='first-k', max_iter=10, 
		tol=0.1, inertia=0.0, frozen=False, random_state=None, verbose=False):
		super().__init__()
		self.name = "KMeans"

		self.centroids = _check_parameter(_cast_as_tensor(centroids, 
			dtype=torch.float32), "centroids", ndim=2)
		self.k = _check_parameter(_cast_as_tensor(k), "k", ndim=0, min_value=2,
			dtypes=(int, torch.int32, torch.int64))

		self.init = _check_parameter(init, "init", value_set=("random", 
			"first-k", "submodular-facility-location", 
			"submodular-feature-based"), ndim=0, dtypes=(str,))
		self.max_iter = _check_parameter(_cast_as_tensor(max_iter), "max_iter",
			ndim=0, min_value=1, dtypes=(int, torch.int32, torch.int64))
		self.tol = _check_parameter(_cast_as_tensor(tol), "tol", ndim=0,
			min_value=0)
		self.inertia = _check_parameter(_cast_as_tensor(inertia), "inertia",
			ndim=0, min_value=0.0, max_value=1.0)
		self.frozen = _check_parameter(_cast_as_tensor(frozen), "frozen",
			ndim=0, value_set=(True, False))
		self.random_state = random_state
		self.verbose = _check_parameter(verbose, "verbose", 
			value_set=(True, False))

		if self.k is None and self.centroids is None:
			raise ValueError("Must specify one of `k` or `centroids`.")

		self.k = len(centroids) if centroids else self.k
		self.d = len(centroids[0]) if centroids else None
		self._initialized = centroids is not None
		self._reset_cache()

	def _initialize(self, X):
		X = _check_parameter(_cast_as_tensor(X), "X", ndim=2)
		self.centroids = _initialize_centroids(X, self.k, algorithm=self.init,
			random_state=self.random_state)

		self.d = X.shape[1]
		self._initialized = True
		self._reset_cache()

	def _reset_cache(self):
		if self._initialized == False:
			return

		self._w_sum = torch.zeros(self.k, self.d)
		self._xw_sum = torch.zeros(self.k, self.d)
		self._centroid_sum = torch.sum(self.centroids**2, axis=1).unsqueeze(0)

	def _distances(self, X):
		X = _check_parameter(_cast_as_tensor(X, dtype=torch.float32), "X", 
			ndim=2, shape=(-1, self.d))

		XX = torch.sum(X**2, axis=1).unsqueeze(1)
		Xc = torch.matmul(X, self.centroids.T)
		
		d = torch.clamp(XX - 2*Xc + self._centroid_sum, min=0)
		return torch.sqrt(d)

	def predict(self, X):
		return self._distances(X).argmin(axis=1)

	def summarize(self, X, sample_weight=None):
		if self.frozen:
			return 0

		if not self._initialized:
			self._initialize(X)

		X = _check_parameter(_cast_as_tensor(X, dtype=torch.float32), "X", 
			ndim=2, shape=(-1, self.d))
		sample_weight = _reshape_weights(X, _cast_as_tensor(sample_weight, 
			dtype=torch.float32))

		distances = self._distances(X)
		y_hat = distances.argmin(dim=1).unsqueeze(1).expand(-1, self.d)

		self._w_sum.scatter_add_(0, y_hat, sample_weight)
		self._xw_sum.scatter_add_(0, y_hat, X * sample_weight)
		return distances.min(dim=1).values.sum()

	def from_summaries(self):
		if self.frozen:
			return

		centroids = self._xw_sum / self._w_sum
		_update_parameter(self.centroids, centroids, self.inertia)
		self._reset_cache()

	def fit(self, X, sample_weight=None):
		d_current = None
		for i in range(self.max_iter):
			start_time = time.time()

			d_previous = d_current
			d_current = self.summarize(X, sample_weight=sample_weight)

			if i > 0:
				improvement = d_previous - d_current
				duration = time.time() - start_time

				if self.verbose:
					print("[{}] Improvement: {}, Time: {:4.4}s".format(i, 
						improvement, duration))

				if improvement < self.tol:
					break

			self.from_summaries()

		self._reset_cache()
		return self

	def fit_predict(self, X, sample_weight=None):
		self.fit(X, sample_weight=sample_weight)
		return self.predict(X)