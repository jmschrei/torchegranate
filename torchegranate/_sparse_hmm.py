import math
import numpy
import torch

from ._utils import _cast_as_tensor
from ._utils import _update_parameter
from ._utils import _check_parameter
from ._utils import _check_hmm_inputs

from .distributions._distribution import Distribution

from ._bayes import BayesMixin
from ._base import GraphMixin
from ._base import Node

NEGINF = float("-inf")

_parameter = lambda x: torch.nn.Parameter(x, requires_grad=False)


def _convert_to_sparse_edges(nodes, edges, starts, ends, start, end):
	if len(edges[0]) == 3:
		_edges = edges
	else:
		n = len(edges)
		_edges = []

		for i in range(n):
			for j in range(n):
				if edges[i, j] != 0:
					_edges.append((nodes[i], nodes[j], edges[i, j])) 

	if starts is not None:
		for i in range(n):
			if starts[i] != 0:
				_edges.append((start, nodes[i], starts[i]))

	if ends is not None:
		for i in range(n):
			if ends[i] != 0:
				_edges.append((nodes[i], end, ends[i]))

	return _edges


class _SparseHMM(Distribution):
	"""A hidden Markov model with a sparse transition matrix.

	A hidden Markov model is an extension of a mixture model to sequences by
	including a transition matrix between the elements of the mixture. Each of
	the algorithms for a hidden Markov model are essentially just a revision
	of those algorithms to incorporate this transition matrix.

	This object is a wrapper for a hidden Markov model with a sparse transition
	matrix.

	Separately, there are two ways to instantiate the hidden Markov model. The
	first is by passing in a set of distributions, a dense transition matrix, 
	and optionally start/end probabilities. The second is to initialize the
	object without these and then to add edges using the `add_edge` method
	and to add nodes using the `add_nodes` method. Importantly, the way that
	you choose to initialize the hidden Markov model is independent of the
	implementation that you end up choosing. If you pass in a dense transition
	matrix, this will be converted to a sparse matrix with all the zeros
	dropped if you choose `kind='sparse'`.


	Parameters
	----------
	distributions: tuple or list
		A set of distribution objects. These objects do not need to be
		initialized, i.e., can be "Normal()". 

	edges: numpy.ndarray, torch.Tensor, or None. shape=(k,k)
		A dense transition matrix of probabilities for how each node or
		distribution passed in connects to each other one. This can contain
		many zeroes, and when paired with `kind='sparse'`, will drop those
		elements from the matrix.

	starts: list, numpy.ndarray, torch.Tensor, or None. shape=(k,)
		The probability of starting at each node. If not provided, assumes
		these probabilities are uniform.

	ends: list, numpy.ndarray, torch.Tensor, or None. shape=(k,)
		The probability of ending at each node. If not provided, assumes
		these probabilities are uniform.

	inertia: float, [0, 1], optional
		Indicates the proportion of the update to apply to the parameters
		during training. When the inertia is 0.0, the update is applied in
		its entirety and the previous parameters are ignored. When the
		inertia is 1.0, the update is entirely ignored and the previous
		parameters are kept, equivalently to if the parameters were frozen.

	frozen: bool, optional
		Whether all the parameters associated with this distribution are frozen.
		If you want to freeze individual pameters, or individual values in those
		parameters, you must modify the `frozen` attribute of the tensor or
		parameter directly. Default is False.
	"""

	def __init__(self, nodes, edges, start, end, starts=None, ends=None, 
		max_iter=10, tol=0.1, inertia=0.0, frozen=False):
		super().__init__(inertia=inertia, frozen=frozen)
		self.name = "_SparseHMM"

		self.start = start
		self.end = end

		self.nodes = nodes
		self.edges = _convert_to_sparse_edges(nodes, edges, starts, ends,
			self.start, self.end)

		self.n_nodes = len(self.nodes)
		self.n_edges = len(self.edges)

		self.starts = _parameter(torch.full((self.n_nodes,), NEGINF))
		self.ends = _parameter(torch.full((self.n_nodes,), NEGINF)) 

		self._edge_idx_starts = torch.empty(self.n_edges, dtype=torch.int64)
		self._edge_idx_ends = torch.empty(self.n_edges, dtype=torch.int64)
		self._edge_log_probabilities = torch.empty(self.n_edges, 
			dtype=torch.float64)

		idx = 0
		for ni, nj, probability in self.edges:
			if ni is self.start:
				j = self.nodes.index(nj)
				self.starts[j] = math.log(probability)

			elif nj is self.end:
				i = self.nodes.index(ni)
				self.ends[i] = math.log(probability)

			else:
				i = self.nodes.index(ni)
				j = self.nodes.index(nj)

				self._edge_idx_starts[idx] = i
				self._edge_idx_ends[idx] = j
				self._edge_log_probabilities[idx] = math.log(probability)
				idx += 1

		self.nodes = torch.nn.ModuleList(self.nodes)

		self._edge_idx_starts = _parameter(self._edge_idx_starts[:idx])
		self._edge_idx_ends = _parameter(self._edge_idx_ends[:idx])
		self._edge_log_probabilities = _parameter(
			self._edge_log_probabilities[:idx])
		self.n_edges = idx

		self._reset_cache()

	def _reset_cache(self):
		"""Reset the internally stored statistics.

		This method is meant to only be called internally. It resets the
		stored statistics used to update the model parameters as well as
		recalculates the cached values meant to speed up log probability
		calculations.
		"""

		self._xw_sum = _parameter(torch.zeros(self.n_edges, 
			dtype=torch.float64))

		self._xw_starts_sum = _parameter(torch.zeros(self.n_nodes, 
			dtype=torch.float64))

		self._xw_ends_sum = _parameter(torch.zeros(self.n_nodes, 
			dtype=torch.float64))

	def forward(self, X, priors=None, emissions=None):
		"""Run the forward algorithm on some data.

		Runs the forward algorithm on a batch of sequences. This is not to be
		confused with a "forward pass" when talking about neural networks. The
		forward algorithm is a dynamic programming algorithm that begins at the
		start state and returns the probability, over all paths through the
		model, that result in the alignment of symbol i to node j.

		
		Parameters
		----------
		X: list, numpy.ndarray, torch.Tensor, shape=(-1, length, self.d)
			A set of examples to evaluate. 		

		priors: list, numpy.ndarray, torch.Tensor, shape=(-1, length, self.d)
			Prior probabilities of assigning each symbol to each node. If not
			provided, do not include in the calculations (conceptually
			equivalent to a uniform probability, but without scaling the
			probabilities).

		emissions: list, numpy.ndarray, torch.Tensor, shape=(-1, length, self.k)
			Precalculated emission log probabilities. These are the
			probabilities of each observation under each probability 
			distribution. When running some algorithms it is more efficient
			to precalculate these and pass them into each call.


		Returns
		-------
		f: torch.Tensor, shape=(-1, length, self.k)
			The log probabilities calculated by the forward algorithm.
		"""

		X, priors, emissions = _check_hmm_inputs(self, X, priors, emissions)
		n, k, d = X.shape

		f = torch.zeros(k, n, self.n_nodes, dtype=torch.float64) + float("-inf")
		f[0] = self.starts + emissions[0].T + priors[:, 0]

		for i in range(1, k):
			p = f[i-1, :, self._edge_idx_starts]
			p += self._edge_log_probabilities.expand(n, -1)

			alpha = torch.max(p, dim=1, keepdims=True).values
			p = torch.exp(p - alpha)

			z = torch.zeros_like(f[i])
			z.scatter_add_(1, self._edge_idx_ends.expand(n, -1), p)

			f[i] = alpha + torch.log(z) + emissions[i].T + priors[:, i]

		f = f.permute(1, 0, 2)
		return f

	def backward(self, X, priors=None, emissions=None):
		"""Run the backward algorithm on some data.

		Runs the backward algorithm on a batch of sequences. This is not to be
		confused with a "backward pass" when talking about neural networks. The
		backward algorithm is a dynamic programming algorithm that begins at end
		of the sequence and returns the probability, over all paths through the
		model, that result in the alignment of symbol i to node j, working
		backwards.

		
		Parameters
		----------
		X: list, numpy.ndarray, torch.Tensor, shape=(-1, length, self.d)
			A set of examples to evaluate. 		

		priors: list, numpy.ndarray, torch.Tensor, shape=(-1, length, self.d)
			Prior probabilities of assigning each symbol to each node. If not
			provided, do not include in the calculations (conceptually
			equivalent to a uniform probability, but without scaling the
			probabilities).

		emissions: list, numpy.ndarray, torch.Tensor, shape=(-1, length, self.k)
			Precalculated emission log probabilities. These are the
			probabilities of each observation under each probability 
			distribution. When running some algorithms it is more efficient
			to precalculate these and pass them into each call.


		Returns
		-------
		b: torch.Tensor, shape=(-1, length, self.k)
			The log probabilities calculated by the backward algorithm.
		"""

		X, priors, emissions = _check_hmm_inputs(self, X, priors, emissions)
		n, k, d = X.shape

		b = torch.zeros(k, n, self.n_nodes, dtype=torch.float64) + float("-inf")
		b[-1] = self.ends

		for i in range(k-2, -1, -1):
			p = b[i+1, :, self._edge_idx_ends]
			p += emissions[i+1, self._edge_idx_ends].T + priors[:, i+1, self._edge_idx_ends]
			p += self._edge_log_probabilities.expand(n, -1)

			alpha = torch.max(p, dim=1, keepdims=True).values
			p = torch.exp(p - alpha)

			z = torch.zeros_like(b[i])
			z.scatter_add_(1, self._edge_idx_starts.expand(n, -1), p)

			b[i] = alpha + torch.log(z)

		b = b.permute(1, 0, 2)
		return b

	def forward_backward(self, X, priors=None, emissions=None):
		"""Run the forward-backward algorithm on some data.

		Runs the forward-backward algorithm on a batch of sequences. This
		algorithm combines the best of the forward and the backward algorithm.
		It combines the probability of starting at the beginning of the sequence
		and working your way to each observation with the probability of
		starting at the end of the sequence and working your way backward to it.

		A number of statistics can be calculated using this information. These
		statistics are powerful inference tools but are also used during the
		Baum-Welch training process. 

		
		Parameters
		----------
		X: list, numpy.ndarray, torch.Tensor, shape=(-1, length, self.d)
			A set of examples to evaluate. 		

		priors: list, numpy.ndarray, torch.Tensor, shape=(-1, length, self.d)
			Prior probabilities of assigning each symbol to each node. If not
			provided, do not include in the calculations (conceptually
			equivalent to a uniform probability, but without scaling the
			probabilities).

		emissions: list, numpy.ndarray, torch.Tensor, shape=(-1, length, self.k)
			Precalculated emission log probabilities. These are the
			probabilities of each observation under each probability 
			distribution. When running some algorithms it is more efficient
			to precalculate these and pass them into each call.


		Returns
		-------
		transitions: torch.Tensor, shape=(-1, n_nodes, n_nodes) or (-1, n_edges)
			The expected number of transitions across each edge that occur
			for each example. The returned transitions follow the structure
			of the transition matrix and so will be dense or sparse as
			appropriate.

		emissions: torch.Tensor, shape=(-1, length, n_nodes)
			The posterior probabilities of each observation belonging to each
			state given that one starts at the beginning of the sequence,
			aligns observations across all paths to get to the current
			observation, and then proceeds to align all remaining observations
			until the end of the sequence.

		starts: torch.Tensor, shape=(-1, n_nodes)
			The probabilities of starting at each node given the 
			forward-backward algorithm.

		ends: torch.Tensor, shape=(-1, n_nodes)
			The probabilities of ending at each node given the forward-backward
			algorithm.

		logp: torch.Tensor, shape=(-1,)
			The log probabilities of each sequence given the model.
		"""

		X, priors, emissions = _check_hmm_inputs(self, X, priors, emissions)
		n, k, d = X.shape

		f = self.forward(X, priors=priors, emissions=emissions)
		b = self.backward(X, priors=priors, emissions=emissions)

		logp = torch.logsumexp(f[:, -1] + self.ends, dim=1)

		t = f[:, :-1, self._edge_idx_starts] + b[:, 1:, self._edge_idx_ends]
		t += emissions[1:, self._edge_idx_ends].permute(2, 0, 1) + priors[:, 1:, self._edge_idx_ends]
		t += self._edge_log_probabilities.expand(n, k-1, -1)

		starts = self.starts + emissions[0].T + priors[:, 0] + b[:, 0]
		starts = torch.exp(starts.T - torch.logsumexp(starts, dim=-1)).T

		ends = self.ends + f[:, -1]
		ends = torch.exp(ends.T - torch.logsumexp(ends, dim=-1)).T

		expected_transitions = torch.exp(torch.logsumexp(t, dim=1).T - logp).T

		fb = f + b
		fb = (fb - torch.logsumexp(fb, dim=2).reshape(len(X), -1, 1))
		return expected_transitions, fb, starts, ends, logp

	def summarize(self, X, sample_weight=None, priors=None):
		"""Extract the sufficient statistics from a batch of data.

		This method calculates the sufficient statistics from optionally
		weighted data and adds them to the stored cache. The examples must be
		given in a 2D format. Sample weights can either be provided as one
		value per example or as a 2D matrix of weights for each feature in
		each example.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)
			A set of examples to summarize.

		sample_weight: list, tuple, numpy.ndarray, torch.Tensor, optional
			A set of weights for the examples. This can be either of shape
			(-1, self.d) or a vector of shape (-1,). Default is ones.

		priors: list, numpy.ndarray, torch.Tensor, shape=(-1, length, self.d)
			Prior probabilities of assigning each symbol to each node. If not
			provided, do not include in the calculations (conceptually
			equivalent to a uniform probability, but without scaling the
			probabilities).
		"""

		transitions, emissions, starts, ends, logps = self.forward_backward(X, 
			priors=priors)

		self._xw_starts_sum += torch.sum(starts * sample_weight, dim=0)
		self._xw_ends_sum += torch.sum(ends * sample_weight, dim=0)
		self._xw_sum += torch.sum(transitions * sample_weight, dim=0) 

		X = X.reshape(-1, X.shape[-1])
		emissions = torch.exp(emissions) * sample_weight.unsqueeze(1)
		for i, node in enumerate(self.nodes):
			w = emissions[:, :, i].reshape(-1, 1)
			node.distribution.summarize(X, sample_weight=w)

		return logps

	def from_summaries(self):
		"""Update the model parameters given the extracted statistics.

		This method uses calculated statistics from calls to the `summarize`
		method to update the distribution parameters. Hyperparameters for the
		update are passed in at initialization time.

		Note: Internally, a call to `fit` is just a successive call to the
		`summarize` method followed by the `from_summaries` method.
		"""

		for node in self.nodes:
			node.distribution.from_summaries()

		if self.frozen:
			return

		node_out_count = torch.clone(self._xw_ends_sum)
		for start, count in zip(self._edge_idx_starts, self._xw_sum):
			node_out_count[start] += count

		ends = torch.log(self._xw_ends_sum / node_out_count)
		starts = torch.log(self._xw_starts_sum / self._xw_starts_sum.sum())
		_edge_log_probabilities = torch.empty_like(self._edge_log_probabilities)

		for i in range(self.n_edges):
			t = self._xw_sum[i]
			t_sum = node_out_count[self._edge_idx_starts[i]]
			_edge_log_probabilities[i] = torch.log(t / t_sum)

		_update_parameter(self.ends, ends, inertia=self.inertia)
		_update_parameter(self.starts, starts, inertia=self.inertia)
		_update_parameter(self._edge_log_probabilities, _edge_log_probabilities,
			inertia=self.inertia)
		self._reset_cache()