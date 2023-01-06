# bayesian_network.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

import torch
import itertools

from ._utils import _cast_as_tensor
from ._utils import _update_parameter
from ._utils import _check_parameter
from ._utils import _reshape_weights

from .distributions._distribution import Distribution
from .distributions import Categorical
from .distributions import ConditionalCategorical

class BayesianNetwork(Distribution):
	def __init__(self, distributions, edges, inertia=0.0, frozen=False):
		super().__init__(inertia=inertia, frozen=frozen)

		self.name = "BayesianNetwork"

		self.distributions = _check_parameter(distributions, "distributions",
			dtypes=(list, tuple))
		self.n_categories = _check_parameter(n_categories, "n_categories",
			dtypes=(list, tuple))

		if distributions is not None:
			self.k = len(distributions) - 1
		
		if n_categories is None:
			self.n_categories = [None for i in range(self.k+1)]

		self.d = None
		self._initialized = distributions is not None and distributions[0]._initialized
		self._reset_cache()

	def _initialize(self, d):
		self.distributions = [Categorical(n_categories=self.n_categories[0])]
		for i in range(self.k):
			self.distributions.append(ConditionalCategorical(n_categories=self.n_categories[i+1]))

		self._initialized = True
		super()._initialize(d)

	def _reset_cache(self):
		return

	def log_probability(self, X):
		X = _check_parameter(_cast_as_tensor(X), "X", ndim=3)
		
		logps = torch.zeros(X.shape, device=X.device, dtype=X.dtype)
		for distribution, parents in zip(self.distributions, self.parents):
			logps += distribution.log_probability(X[:, parents].unsqueeze(-1))

		return logps

	def fit(self, X, sample_weight=None):
		self.summarize(X, sample_weight=sample_weight)
		self.from_summaries()
		return self

	def summarize(self, X, sample_weight=None):
		if self.frozen:
			return

		if not self._initialized:
			self._initialize(len(X[0]))

		X = _check_parameter(_cast_as_tensor(X), "X", ndim=3)
		sample_weight = _check_parameter(_cast_as_tensor(sample_weight), 
			"sample_weight", min_value=0, ndim=(1, 2))

		if sample_weight is None:
			sample_weight = torch.ones_like(X[:, 0])
		elif len(sample_weight.shape) == 1: 
			sample_weight = sample_weight.reshape(-1, 1).expand(-1, X.shape[2])
		elif sample_weight.shape[1] == 1:
			sample_weight = sample_weight.expand(-1, X.shape[2])

		_check_parameter(_cast_as_tensor(sample_weight), "sample_weight", 
			min_value=0, ndim=2, shape=(X.shape[0], X.shape[2]))

		for distribution, parents in zip(self.distibutions, self.parents):
			distribution.summarize(X[:, parents].unsqueeze(1))

	def from_summaries(self):
		if self.frozen:
			return

		for distribution in self.distributions:
			distribution.from_summaries()

#####

def _discrete_find_best_parents(X, sample_weight, n_categories, pseudocount, max_parents, parent_set, i):
	best_score, best_parents = float("-inf"), None
	for k in range(min(max_parents, len(parent_set))+1):
		for parents in itertools.combinations(parent_set, k):
			columns = list(parents) + [i]
			n_categories_ = tuple(n_categories[columns]) if k > 0 else (i,)

			score = _discrete_score_node(X[:, columns], sample_weight, n_categories_,
				pseudocount)

			if score > best_score:
				best_score = score
				best_parents = parents

	return best_score, best_parents

def _discrete_score_node(X, sample_weight, n_categories, pseudocount):
	counts = torch.zeros(*n_categories) + pseudocount

	for x, w in zip(X, sample_weight):
		x = tuple(x)
		counts[x] += w.squeeze()

	marginal_counts = counts.sum(dim=-1, keepdims=True)

	logp = torch.sum(counts * torch.log(counts / marginal_counts))
	logp -= torch.log(sample_weight.sum()) / 2 + torch.prod(torch.tensor(n_categories))
	return logp
