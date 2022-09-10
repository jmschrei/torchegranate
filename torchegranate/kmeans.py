# kmeans.py
# Author: Jacob Schreiber

import torch
import time

from apricot import FacilityLocationSelection
from apricot import FeatureBasedSelection

def _initialize_centroids(X, k, algorithm='first-k'):
	if algorithm == 'first-k':
		return torch.clone(X[:k])

	elif algorithm == 'random':
		idxs = torch.arange(X.shape[0])
		torch.shuffle(idxs)
		return torch.clone(X[idxs[:k]])

	elif algorithm == 'submodular-facility-location':
		selector = FacilityLocationSelection(k)
		a = selector.fit_transform(X)
		print(type(a), a.shape)
		dawdwadaw

	elif algorithm == 'submodular-feature-based':
		selector = FeatureBasedSelection(k)
		return selector.fit_transform(X)

	elif algorithm == 'KMeans':
		selector = KMeans(k=k)
		selector.fit(X)
		return torch.clone(selector.centroids) 



class KMeans():
	def __init__(self, k=None, centroids=None, init='random', stop_threshold=0.01, max_iters=100, frozen=True):
		self.k = k
		self.centroids = centroids

		if centroids is None:
			self._initialized = False
			self.d = None
		else:
			self._initialized = True
			self.d = centroids.shape[1]

		self.init = init
		self.stop_threshold = stop_threshold
		self.max_iters = max_iters
		self.frozen = frozen

		self._reset_cache()


	def _reset_cache(self):
		if self._initialized:
			self._x_sum = torch.zeros(self.k, self.d)
			self._w_sum = torch.zeros(self.k)
		else:
			self._x_sum = None
			self._w_sum = None


	def _initialize(self, X):
		self.centroids = _initialize_centroids(X, self.k, algorithm=self.init)
		self.d = X.shape[1]
		self._initialized = True
		self._reset_cache()

	def _distances(self, X):
		d1 = torch.sum(X**2, axis=1).unsqueeze(1)
		d2 = torch.matmul(X, self.centroids.T)
		d3 = torch.sum(self.centroids**2, axis=1).unsqueeze(0)
		return torch.sqrt(d1 - 2*d2 + d3)

	def predict(self, X):
		return self._distances(X).argmin(axis=1)

	def summarize(self, X, sample_weights=None):
		if not self._initialized:
			self._initialize(X)

		if sample_weights is None:
			sample_weights = torch.ones(X.shape[0])

		y_hat = self.predict(X)

		for i in range(self.k):
			self._x_sum[i] += X[y_hat == i].sum(axis=0)
			self._w_sum[i] += sample_weights[y_hat == i].sum()

	def from_summaries(self):
		self.centroids = self._x_sum / self._w_sum.unsqueeze(1)
		self._reset_cache()

	def fit(self, X, sample_weights=None):
		self._initialize(X)
		d_initial = self._distances(X).min(axis=1).values.sum()
		d_previous = d_initial

		for i in range(self.max_iters):
			self.summarize(X)
			self.from_summaries()

			d_current = self._distances(X).min(axis=1).values.sum()

			if d_previous - d_current < self.stop_threshold:
				break

		return self


n = 1000
d = 20
k = 3

xs = []
for i in range(k):
	x = torch.randn(n, d) + i
	xs.append(x)

X = torch.cat(xs)
centroids = torch.stack([x.mean(axis=0) for x in xs])

tic = time.time()
model = KMeans(k=k, init='submodular-facility-location').fit(X)
toc1 = time.time() - tic

toc2 = 0

print(toc1, toc2)

print(model.predict(X))