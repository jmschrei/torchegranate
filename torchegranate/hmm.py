# hmm.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

import time
import numpy
import torch

from ._utils import _cast_as_tensor
from ._utils import _update_parameter
from ._utils import _check_parameter
from ._utils import _reshape_weights

from .distributions._distribution import Distribution

from ._bayes import BayesMixin
from ._base import GraphMixin
from ._base import Node

from ._sparse_hmm import _SparseHMM
from ._dense_hmm import _DenseHMM

from .kmeans import KMeans


NEGINF = float("-inf")

_parameter = lambda x: torch.nn.Parameter(x, requires_grad=False)


def _cast_distributions(distributions):
	if distributions is None:
		return []

	nodes = []
	for i, distribution in enumerate(distributions):
		if isinstance(distribution, Node):
			nodes.append(distribution)
		elif isinstance(distribution, Distribution):
			nodes.append(Node(distribution, str(i)))
		else:
			raise ValueError("Nodes must be node or distribution objects.")

	return nodes


class HiddenMarkovModel(GraphMixin, Distribution):
	def __init__(self, nodes=None, edges=None, starts=None, ends=None, 
		kind="sparse", init='random', batch_size=2048, max_iter=10, tol=0.1, 
		inertia=0.0, frozen=False, random_state=None, verbose=False):
		super().__init__(inertia=inertia, frozen=frozen)
		self.name = "HiddenMarkovModel"

		_check_parameter(kind, "kind", value_set=('sparse', 'dense'))

		self.nodes = _cast_distributions(nodes)

		n = len(nodes) if nodes is not None else None
		self.n_nodes = n
		self.n_edges = len(edges) if edges is not None else None

		self.edges = _check_parameter(_cast_as_tensor(edges), "edges",
			ndim=2, shape=(n, n), min_value=0., max_value=1.)
		self.starts = _check_parameter(_cast_as_tensor(starts), "starts",
			ndim=1, shape=(n,), min_value=0., max_value=1., value_sum=1.0)
		self.ends = _check_parameter(_cast_as_tensor(ends), "ends",
			ndim=1, shape=(n,), min_value=0., max_value=1.)

		if self.edges is None and nodes is not None:
			self.edges = torch.ones(self.n_nodes, self.n_nodes) / self.n_nodes
		elif self.edges is None and nodes is None:
			self.edges = []


		if self.starts is None and nodes is not None:
			self.starts = torch.ones(self.n_nodes) / self.n_nodes

		if self.ends is None and nodes is not None:
			self.ends = torch.ones(self.n_nodes) / self.n_nodes


		self.start = Node(None, "start")
		self.end = Node(None, "end")

		self.kind = kind
		self.init = init
		self.batch_size = _check_parameter(batch_size, "batch_size", 
			min_value=1, ndim=0)
		self.max_iter = _check_parameter(max_iter, "max_iter", min_value=1, 
			ndim=0, dtypes=(int, torch.int32, torch.int64))
		self.tol = _check_parameter(tol, "tol", min_value=0., ndim=0)
		self.random_state = random_state
		self.verbose = verbose

		self.d = self.nodes[0].distribution.d if nodes is not None else None
		self._model = None
		self._initialized = False


	@property
	def device(self):
		return self._model.device

	def bake(self):
		if self.kind == 'dense':
			self._model = _DenseHMM(nodes=self.nodes, edges=self.edges,
				start=self.start, end=self.end, starts=self.starts, 
				ends=self.ends, batch_size=self.batch_size,
				max_iter=self.max_iter, tol=self.tol, inertia=self.inertia, 
				frozen=self.frozen)

		elif self.kind == 'sparse':
			self._model = _SparseHMM(nodes=self.nodes, edges=self.edges,
				start=self.start, end=self.end, starts=self.starts, 
				ends=self.ends, batch_size=self.batch_size,
				max_iter=self.max_iter, tol=self.tol, inertia=self.inertia, 
				frozen=self.frozen)

		self.n_nodes = self._model.n_nodes
		self.n_edges = self._model.n_edges
		self._initialized = True

	def _reset_cache(self):
		self._model._reset_cache()
		for node in self.nodes:
			node.distribution._reset_cache()

		if self.kind == 'sparse':
			self.edges = self._model._edge_log_probabilities
		else:
			self.edges = self._model.edges

		self.starts = self._model.starts
		self.ends = self._model.ends

	def _initialize(self, X):
		X = _check_parameter(_cast_as_tensor(X), "X", ndim=3)
		X = X.reshape(-1, X.shape[-1])

		y_hat = KMeans(self.n_nodes, init=self.init, max_iter=3, 
			random_state=self.random_state).fit_predict(X)

		for i in range(self.n_nodes):
			self.nodes[i].distribution.fit(X[y_hat == i])

		self._initialized = True
		self._reset_cache()
		self.d = X.shape[-1]
		super()._initialize(X.shape[-1])

	def _emission_matrix(self, X):
		X = _check_parameter(_cast_as_tensor(X), "X", ndim=3, 
			shape=(-1, -1, self.d))

		n, k, _ = X.shape
		X = X.reshape(-1, self.d)

		e = torch.empty((k, self.n_nodes, n), dtype=torch.float64)
		for i, node in enumerate(self.nodes):
			e[:, i] = node.distribution.log_probability(X).reshape(n, k).T

		return e.permute(2, 0, 1)

	def forward(self, X, priors=None, emissions=None):
		return self._model.forward(X, priors=priors, emissions=emissions)

	def backward(self, X, priors=None, emissions=None):
		return self._model.backward(X, priors=priors, emissions=emissions)

	def forward_backward(self, X, priors=None, emissions=None):
		return self._model.forward_backward(X, priors=priors, 
			emissions=emissions)

	def log_probability(self, X, priors=None):
		f = self.forward(X, priors=priors)
		return torch.logsumexp(f[:, -1] + self.ends, dim=1)

	def predict_log_proba(self, X, priors=None):
		_, fb, _, _, _ = self._model.forward_backward(X, priors=priors)
		return fb

	def predict_proba(self, X, priors=None):
		return torch.exp(self.predict_log_proba(X, priors=priors))

	def predict(self, X, priors=None):
		return torch.argmax(self.predict_log_proba(X, priors=priors), dim=-1)

	def fit(self, X, sample_weight=None, priors=None):
		logp = None
		for i in range(self.max_iter):
			start_time = time.time()
			logp = self.summarize(X, sample_weight=sample_weight).sum()

			if i > 0:
				improvement = logp - last_logp
				duration = time.time() - start_time

				if self.verbose:
					print("[{}] Improvement: {}, Time: {:4.4}s".format(i, 
						improvement, duration))

				if improvement < self.tol:
					break

			last_logp = logp
			self.from_summaries()

		if self.verbose:
			logp = self.summarize(X, sample_weight=sample_weight).sum()

			improvement = logp - last_logp
			duration = time.time() - start_time

			print("[{}] Improvement: {}, Time: {:4.4}s".format(i+1, 
				improvement, duration))

		self._reset_cache()
		return self

	def summarize(self, X, sample_weight=None, priors=None):
		X = _check_parameter(_cast_as_tensor(X), "X", ndim=3)
		
		if sample_weight is None:
			sample_weight = torch.ones(1).expand(X.shape[0], 1)
		else:
			sample_weight = _check_parameter(_cast_as_tensor(sample_weight),
				"sample_weight", min_value=0., ndim=1, 
				shape=(len(X),)).reshape(-1, 1)

		return self._model.summarize(X, sample_weight=sample_weight, 
			priors=priors)

	def from_summaries(self):
		self._model.from_summaries()
		self._reset_cache()
