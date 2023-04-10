# _sparse_hmm.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

import math
import numpy
import torch

from ._utils import _cast_as_tensor
from ._utils import _cast_as_parameter
from ._utils import _update_parameter
from ._utils import _check_parameter

from .distributions._distribution import Distribution

from ._base import _BaseHMM
from ._base import _check_inputs

NEGINF = float("-inf")
inf = float("inf")


def _convert_to_sparse_edges(distributions, edges, starts, ends, start, end):
	n = len(distributions)

	if len(edges[0]) == 3 and len(distributions) != 3:
		_edges = edges
	else:
		n = len(edges)
		_edges = []

		for i in range(n):
			for j in range(n):
				if edges[i, j] != 0:
					_edges.append((distributions[i], distributions[j], edges[i, j])) 

	if starts is not None:
		for i in range(n):
			if starts[i] != 0:
				_edges.append((start, distributions[i], starts[i]))

	if ends is not None:
		for i in range(n):
			if ends[i] != 0:
				_edges.append((distributions[i], end, ends[i]))

	return _edges


class SparseHMM(_BaseHMM):
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
	and to add distributions using the `add_distributions` method. Importantly, the way that
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

	def __init__(self, distributions=None, edges=None, starts=None, ends=None, 
		init='random', max_iter=1000, tol=0.1, sample_length=None, 
		return_sample_paths=False, inertia=0.0, frozen=False, random_state=None, 
		verbose=False):
		super().__init__(distributions=distributions, edges=edges, 
			starts=starts, ends=ends, init=init, max_iter=max_iter, tol=tol,
			sample_length=sample_length, 
			return_sample_paths=return_sample_paths, inertia=inertia,
			frozen=frozen, random_state=random_state, verbose=verbose)
		self.name = "SparseHMM"

		self.distributions = torch.nn.ModuleList(distributions)
		self.edges = _convert_to_sparse_edges(distributions, self.edges, 
			self.starts, self.ends, self.start, self.end)
		self.n_edges = len(self.edges)

		n = self.n_distributions

		if torch.isinf(self.starts).sum() == len(self.starts):
			self.starts = _cast_as_parameter(torch.ones(n) / n)
		if torch.isinf(self.ends).sum() == len(self.ends):
			self.ends = _cast_as_parameter(torch.ones(n) / n)

		_edge_idx_starts = _cast_as_parameter(torch.empty(self.n_edges, 
			dtype=torch.int64))
		_edge_idx_ends = _cast_as_parameter(torch.empty(self.n_edges, 
			dtype=torch.int64))
		_edge_log_probs = _cast_as_parameter(torch.empty(self.n_edges, 
			dtype=torch.float32))

		idx = 0
		for ni, nj, probability in self.edges:
			if ni is self.start:
				j = distributions.index(nj)
				self.starts[j] = math.log(probability)

			elif nj is self.end:
				i = distributions.index(ni)
				self.ends[i] = math.log(probability)

			else:
				i = distributions.index(ni)
				j = distributions.index(nj)

				_edge_idx_starts[idx] = i
				_edge_idx_ends[idx] = j
				_edge_log_probs[idx] = math.log(probability)
				idx += 1

		if torch.isinf(self.starts).sum() == len(self.starts):
			self.starts = torch.ones(n) / n
		if torch.isinf(self.ends).sum() == len(self.ends):
			self.ends = torch.ones(n) / n

		self._edge_idx_starts = _cast_as_parameter(_edge_idx_starts[:idx])
		self._edge_idx_ends = _cast_as_parameter(_edge_idx_ends[:idx])
		self._edge_log_probs = _cast_as_parameter(_edge_log_probs[:idx])
		self.n_edges = idx

		self._edge_keymap = {}
		for i in range(idx):
			start = self._edge_idx_starts[i].item()
			end = self._edge_idx_ends[i].item()
			self._edge_keymap[(start, end)] = i

		self.edges = self._edge_log_probs

		self._reset_cache()

	def bake(self):
		return

	def _reset_cache(self):
		"""Reset the internally stored statistics.

		This method is meant to only be called internally. It resets the
		stored statistics used to update the model parameters as well as
		recalculates the cached values meant to speed up log probability
		calculations.
		"""

		if self._initialized == False:
			return

		for node in self.distributions:
			node._reset_cache()

		self.register_buffer("_xw_sum", torch.zeros(self.n_edges, 
			dtype=self.dtype, device=self.device))

		self.register_buffer("_xw_starts_sum", torch.zeros(
			self.n_distributions, dtype=self.dtype, device=self.device))

		self.register_buffer("_xw_ends_sum", torch.zeros(
			self.n_distributions, dtype=self.dtype, device=self.device))

	def sample(self, n):
		"""Sample from the probability distribution.

		This method will return `n` samples generated from the underlying
		probability distribution. Because a HMM describes variable length
		sequences, a list will be returned where each element is one of
		the generated sequences.


		Parameters
		----------
		n: int
			The number of samples to generate.
		

		Returns
		-------
		X: list of torch.tensor, shape=(n,)
			A list of randomly generated samples, where each sample of
			size (length, self.d).
		"""

		if self.sample_length is None and self.ends is None:
			raise ValueError("Must specify a length or have explicit "
				+ "end probabilities.")

		distributions, emissions = [], []
		edge_ends, edge_probs = [], []
		for idx in range(self.n_distributions):
			idxs = self._edge_idx_starts == idx

			_ends = numpy.concatenate([self._edge_idx_ends[idxs].numpy(), 
				[self.n_distributions]])
			_probs = numpy.concatenate([torch.exp(self._edge_log_probs[idxs]
				).numpy(), [numpy.exp(self.ends[idx])]])

			edge_ends.append(_ends)
			edge_probs.append(_probs)

		starts = torch.exp(self.starts).numpy()

		for _ in range(n):	
			node_i = self.random_state.choice(self.n_distributions, p=starts)
			emission_i = self.distributions[node_i].sample(n=1)
			distributions_, emissions_ = [node_i], [emission_i]

			for i in range(1, self.sample_length or int(1e8)):
				node_i = self.random_state.choice(edge_ends[node_i], p=edge_probs[node_i])
				if node_i == self.n_distributions:
					break

				emission_i = self.distributions[node_i].sample(n=1)

				distributions_.append(node_i)
				emissions_.append(emission_i)

			distributions.append(distributions_)
			emissions.append(torch.vstack(emissions_))

		if self.return_sample_paths == True:
			return emissions, distributions
		return emissions

	def forward(self, X=None, emissions=None, priors=None, check_inputs=True):
		"""Run the forward algorithm on some data.

		Runs the forward algorithm on a batch of sequences. This is not to be
		confused with a "forward pass" when talking about neural networks. The
		forward algorithm is a dynamic programming algorithm that begins at the
		start state and returns the probability, over all paths through the
		model, that result in the alignment of symbol i to node j.

		Note that, as an internal method, this does not take as input the
		actual sequence of observations but, rather, the emission probabilities
		calculated from the sequence given the model.

		
		Parameters
		----------
		X: list, numpy.ndarray, torch.Tensor, shape=(-1, -1, d)
			A set of examples to evaluate. Does not need to be passed in if
			emissions are. 

		emissions: list, numpy.ndarray, torch.Tensor, shape=(-1, -1, n_distributions)
			Precalculated emission log probabilities. These are the
			probabilities of each observation under each probability 
			distribution. When running some algorithms it is more efficient
			to precalculate these and pass them into each call.

		priors: list, numpy.ndarray, torch.Tensor, shape=(-1, -1, d)
			Prior probabilities of assigning each symbol to each node. If not
			provided, do not include in the calculations (conceptually
			equivalent to a uniform probability, but without scaling the
			probabilities).

		check_inputs: bool, optional
			Whether to check the shape of the inputs and calculate emission
			matrices. Default is True.

		Returns
		-------
		f: torch.Tensor, shape=(-1, -1, self.n_distributions)
			The log probabilities calculated by the forward algorithm.
		"""

		emissions, priors = _check_inputs(self, X, emissions, priors)
		n, l, _ = emissions.shape

		f = torch.full((l, n, self.n_distributions), -inf, dtype=torch.float32, 
			device=self.device)
		f[0] = self.starts + emissions[:, 0] + priors[:, 0]

		for i in range(1, l):
			p = f[i-1, :, self._edge_idx_starts]
			p += self._edge_log_probs.expand(n, -1)

			alpha = torch.max(p, dim=1, keepdims=True).values
			p = torch.exp(p - alpha)

			z = torch.zeros_like(f[i])
			z.scatter_add_(1, self._edge_idx_ends.expand(n, -1), p)

			f[i] = alpha + torch.log(z) + emissions[:, i] + priors[:, i]

		f = f.permute(1, 0, 2)
		return f

	def backward(self, X=None, emissions=None, priors=None, check_inputs=True):
		"""Run the backward algorithm on some data.

		Runs the backward algorithm on a batch of sequences. This is not to be
		confused with a "backward pass" when talking about neural networks. The
		backward algorithm is a dynamic programming algorithm that begins at end
		of the sequence and returns the probability, over all paths through the
		model, that result in the alignment of symbol i to node j, working
		backwards.

		Note that, as an internal method, this does not take as input the
		actual sequence of observations but, rather, the emission probabilities
		calculated from the sequence given the model.

		
		Parameters
		----------
		X: list, numpy.ndarray, torch.Tensor, shape=(-1, -1, d)
			A set of examples to evaluate. Does not need to be passed in if
			emissions are. 

		emissions: list, numpy.ndarray, torch.Tensor, shape=(-1, -1, n_distributions)
			Precalculated emission log probabilities. These are the
			probabilities of each observation under each probability 
			distribution. When running some algorithms it is more efficient
			to precalculate these and pass them into each call.

		priors: list, numpy.ndarray, torch.Tensor, shape=(-1, -1, d)
			Prior probabilities of assigning each symbol to each node. If not
			provided, do not include in the calculations (conceptually
			equivalent to a uniform probability, but without scaling the
			probabilities).

		check_inputs: bool, optional
			Whether to check the shape of the inputs and calculate emission
			matrices. Default is True.


		Returns
		-------
		b: torch.Tensor, shape=(-1, length, self.n_distributions)
			The log probabilities calculated by the backward algorithm.
		"""

		emissions, priors = _check_inputs(self, X, emissions, priors)
		n, l, _ = emissions.shape

		b = torch.full((l, n, self.n_distributions), -inf, dtype=torch.float32,
			device=self.device)
		b[-1] = self.ends

		for i in range(l-2, -1, -1):
			p = b[i+1, :, self._edge_idx_ends]
			p += emissions[:, i+1, self._edge_idx_ends] + priors[:, i+1, self._edge_idx_ends]
			p += self._edge_log_probs.expand(n, -1)

			alpha = torch.max(p, dim=1, keepdims=True).values
			p = torch.exp(p - alpha)

			z = torch.zeros_like(b[i])
			z.scatter_add_(1, self._edge_idx_starts.expand(n, -1), p)

			b[i] = alpha + torch.log(z)

		b = b.permute(1, 0, 2)
		return b

	def forward_backward(self, X=None, emissions=None, priors=None, 
		check_inputs=True):
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
		X: list, numpy.ndarray, torch.Tensor, shape=(-1, -1, d)
			A set of examples to evaluate. Does not need to be passed in if
			emissions are. 

		emissions: list, numpy.ndarray, torch.Tensor, shape=(-1, -1, n_distributions)
			Precalculated emission log probabilities. These are the
			probabilities of each observation under each probability 
			distribution. When running some algorithms it is more efficient
			to precalculate these and pass them into each call.

		priors: list, numpy.ndarray, torch.Tensor, shape=(-1, -1, d)
			Prior probabilities of assigning each symbol to each node. If not
			provided, do not include in the calculations (conceptually
			equivalent to a uniform probability, but without scaling the
			probabilities).

		check_inputs: bool, optional
			Whether to check the shape of the inputs and calculate emission
			matrices. Default is True.


		Returns
		-------
		transitions: torch.Tensor, shape=(-1, n, n)
			The expected number of transitions across each edge that occur
			for each example. The returned transitions follow the structure
			of the transition matrix and so will be dense or sparse as
			appropriate.

		responsibility: torch.Tensor, shape=(-1, -1, n)
			The posterior probabilities of each observation belonging to each
			state given that one starts at the beginning of the sequence,
			aligns observations across all paths to get to the current
			observation, and then proceeds to align all remaining observations
			until the end of the sequence.

		starts: torch.Tensor, shape=(-1, n)
			The probabilities of starting at each node given the 
			forward-backward algorithm.

		ends: torch.Tensor, shape=(-1, n)
			The probabilities of ending at each node given the forward-backward
			algorithm.

		logp: torch.Tensor, shape=(-1,)
			The log probabilities of each sequence given the model.
		"""

		emissions, priors = _check_inputs(self, X, emissions, priors)
		n, l, _ = emissions.shape

		f = self.forward(emissions=emissions, priors=priors)
		b = self.backward(emissions=emissions, priors=priors)

		logp = torch.logsumexp(f[:, -1] + self.ends, dim=1)

		t = f[:, :-1, self._edge_idx_starts] + b[:, 1:, self._edge_idx_ends]
		t += emissions[:, 1:, self._edge_idx_ends] + priors[:, 1:, self._edge_idx_ends]
		t += self._edge_log_probs.expand(n, l-1, -1)
		t = torch.exp(torch.logsumexp(t, dim=1).T - logp).T

		starts = self.starts + emissions[:, 0] + priors[:, 0] + b[:, 0]
		starts = torch.exp(starts.T - torch.logsumexp(starts, dim=-1)).T

		ends = self.ends + f[:, -1]
		ends = torch.exp(ends.T - torch.logsumexp(ends, dim=-1)).T

		r = f + b
		r = (r - torch.logsumexp(r, dim=2).reshape(n, -1, 1))
		return t, r, starts, ends, logp

	def summarize(self, X, sample_weight=None, emissions=None, priors=None):
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


		emissions: torch.Tensor, shape=(-1, -1, self.n_distributions)
			Precalculated emission log probabilities. These are the
			probabilities of each observation under each probability 
			distribution. When running some algorithms it is more efficient
			to precalculate these and pass them into each call.	

		priors: list, numpy.ndarray, torch.Tensor, shape=(-1, length, self.d)
			Prior probabilities of assigning each symbol to each node. If not
			provided, do not include in the calculations (conceptually
			equivalent to a uniform probability, but without scaling the
			probabilities).
		"""

		X, emissions, priors, sample_weight = super().summarize(X, 
			sample_weight=sample_weight, emissions=emissions, priors=priors)

		t, r, starts, ends, logps = self.forward_backward(emissions=emissions, 
			priors=priors)

		self._xw_starts_sum += torch.sum(starts * sample_weight, dim=0)
		self._xw_ends_sum += torch.sum(ends * sample_weight, dim=0)
		self._xw_sum += torch.sum(t * sample_weight, dim=0) 

		X = X.reshape(-1, X.shape[-1])
		r = torch.exp(r) * sample_weight.unsqueeze(1)
		for i, node in enumerate(self.distributions):
			w = r[:, :, i].reshape(-1, 1)
			node.summarize(X, sample_weight=w)

		return logps

	def from_summaries(self):
		"""Update the model parameters given the extracted statistics.

		This method uses calculated statistics from calls to the `summarize`
		method to update the distribution parameters. Hyperparameters for the
		update are passed in at initialization time.

		Note: Internally, a call to `fit` is just a successive call to the
		`summarize` method followed by the `from_summaries` method.
		"""

		for node in self.distributions:
			node.from_summaries()

		if self.frozen:
			return

		node_out_count = torch.clone(self._xw_ends_sum)
		for start, count in zip(self._edge_idx_starts, self._xw_sum):
			node_out_count[start] += count

		ends = torch.log(self._xw_ends_sum / node_out_count)
		starts = torch.log(self._xw_starts_sum / self._xw_starts_sum.sum())
		_edge_log_probs = torch.empty_like(self._edge_log_probs)

		for i in range(self.n_edges):
			t = self._xw_sum[i]
			t_sum = node_out_count[self._edge_idx_starts[i]]
			_edge_log_probs[i] = torch.log(t / t_sum)

		_update_parameter(self.ends, ends, inertia=self.inertia)
		_update_parameter(self.starts, starts, inertia=self.inertia)
		_update_parameter(self._edge_log_probs, _edge_log_probs,
			inertia=self.inertia)
		self._reset_cache()