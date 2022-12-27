import math
import numpy
import torch

from ._utils import _cast_as_tensor
from ._utils import _cast_as_parameter
from ._utils import _update_parameter
from ._utils import _check_parameter

from .distributions._distribution import Distribution

from ._bayes import BayesMixin
from ._base import GraphMixin
from ._base import Node

NEGINF = float("-inf")
inf = float("inf")

def _convert_to_dense_edges(nodes, edges, starts, ends, start, end):
	n = len(nodes)

	if len(edges[0]) == n:
		edges = _cast_as_parameter(torch.log(_cast_as_tensor(edges, 
			dtype=torch.float32)))
		starts = _cast_as_parameter(torch.log(_cast_as_tensor(starts,
			dtype=torch.float32)))
		ends = _cast_as_parameter(torch.log(_cast_as_tensor(ends,
			dtype=torch.float32)))
		return edges, starts, ends

	else:
		starts = _cast_as_parameter(torch.full(n, -inf))
		ends = _cast_as_parameter(torch.full(n, -inf))
		_edges = _cast_as_parameter(torch.full((n, n), -inf))
		for ni, nj, probability in edges:
			if ni == start:
				j = nodes.index(nj)
				starts[j] = math.log(probability)
			
			elif nj == end:
				i = nodes.index(ni)
				ends[i] = math.log(probability)

			else:
				i = nodes.index(ni)
				j = nodes.index(nj)

				_edges[i, j] = math.log(probability)

		return _edges, starts, ends


class _DenseHMM(Distribution):
	"""A hidden Markov model with a dense transition matrix.

	A hidden Markov model is an extension of a mixture model to sequences by
	including a transition matrix between the elements of the mixture. Each of
	the algorithms for a hidden Markov model are essentially just a revision
	of those algorithms to incorporate this transition matrix.

	This object is a wrapper for a hidden Markov model with a dense transition
	matrix.

	This object is a wrapper for both implementations, which can be specified
	using the `kind` parameter. Choosing the right implementation will not
	effect the accuracy of the results but will change the speed at which they
	are calculated. 	

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

	edges: numpy.ndarray, torch.Tensor, or None. shape=(k,k), optional
		A dense transition matrix of probabilities for how each node or
		distribution passed in connects to each other one. This can contain
		many zeroes, and when paired with `kind='sparse'`, will drop those
		elements from the matrix.

	starts: list, numpy.ndarray, torch.Tensor, or None. shape=(k,), optional
		The probability of starting at each node. If not provided, assumes
		these probabilities are uniform.

	ends: list, numpy.ndarray, torch.Tensor, or None. shape=(k,), optional
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
		self.name = "_DenseHMM"

		self.start = start
		self.end = end

		self.nodes = torch.nn.ModuleList(nodes)
		self.edges, self.starts, self.ends = _convert_to_dense_edges(nodes, 
			edges, starts, ends, self.start, self.end)

		self.n_nodes = len(nodes)
		self.n_edges = len(edges)

		if torch.isinf(self.starts).sum() == len(self.starts):
			self.starts = _cast_as_parameter(torch.ones(self.n_nodes) 
				/ self.n_nodes)
		if torch.isinf(self.ends).sum() == len(self.ends):
			self.ends = _cast_as_parameter(torch.ones(self.n_nodes) 
				/ self.n_nodes)

		self._reset_cache()

	def _reset_cache(self):
		"""Reset the internally stored statistics.

		This method is meant to only be called internally. It resets the
		stored statistics used to update the model parameters as well as
		recalculates the cached values meant to speed up log probability
		calculations.
		"""

		self.register_buffer("_xw_sum", torch.zeros(self.n_nodes, self.n_nodes, 
			dtype=torch.float32, requires_grad=False, device=self.device))

		self.register_buffer("_xw_starts_sum", torch.zeros(self.n_nodes, 
			dtype=torch.float32, requires_grad=False, device=self.device))

		self.register_buffer("_xw_ends_sum", torch.zeros(self.n_nodes, 
			dtype=torch.float32, requires_grad=False, device=self.device))


	@torch.inference_mode()
	def forward(self, emissions, priors):
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
		emissions: torch.Tensor, shape=(-1, -1, self.n_nodes)
			Precalculated emission log probabilities. These are the
			probabilities of each observation under each probability 
			distribution. When running some algorithms it is more efficient
			to precalculate these and pass them into each call. 		

		priors: torch.Tensor, shape=(-1, -1, self.n_nodes)
			Prior probabilities of assigning each symbol to each node. If not
			provided, do not include in the calculations (conceptually
			equivalent to a uniform probability, but without scaling the
			probabilities).


		Returns
		-------
		f: torch.Tensor, shape=(-1, -1, self.n_nodes)
			The log probabilities calculated by the forward algorithm.
		"""

		l = emissions.shape[1]

		t_max = self.edges.max()
		t = torch.exp(self.edges - t_max)
		f = torch.clone(emissions.permute(1, 0, 2)).contiguous()
		f[0] += self.starts + priors[:, 0]
		f[1:] += t_max

		for i in range(1, l):
			p_max = torch.max(f[i-1], dim=1, keepdims=True).values
			p = torch.exp(f[i-1] - p_max)
			f[i] += torch.log(torch.matmul(p, t)) + p_max

		f = f.permute(1, 0, 2)
		return f

	@torch.inference_mode()
	def backward(self, emissions, priors):
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
		emissions: torch.Tensor, shape=(-1, l, self.n_nodes)
			Precalculated emission log probabilities. These are the
			probabilities of each observation under each probability 
			distribution. When running some algorithms it is more efficient
			to precalculate these and pass them into each call.

		priors: torch.Tensor, shape=(-1, l, self.n_nodes)
			Prior probabilities of assigning each symbol to each node. If not
			provided, do not include in the calculations (conceptually
			equivalent to a uniform probability, but without scaling the
			probabilities).


		Returns
		-------
		b: torch.Tensor, shape=(-1, length, self.n_nodes)
			The log probabilities calculated by the backward algorithm.
		"""

		n, l, _ = emissions.shape

		b = torch.zeros(l, n, self.n_nodes, dtype=torch.float32, device=self.device) + float("-inf")
		b[-1] = self.ends

		t_max = self.edges.max()
		t = torch.exp(self.edges.T - t_max)

		for i in range(l-2, -1, -1):
			p = b[i+1] + emissions[:, i+1]
			p_max = torch.max(p, dim=1, keepdims=True).values
			p = torch.exp(p - p_max)

			b[i] = torch.log(torch.matmul(p, t)) + t_max + p_max

		b = b.permute(1, 0, 2)
		return b

	@torch.inference_mode()
	def forward_backward(self, emissions, priors):
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
		emissions: torch.Tensor, shape=(-1, -1, self.n_nodes)
			Precalculated emission log probabilities. These are the
			probabilities of each observation under each probability 
			distribution. When running some algorithms it is more efficient
			to precalculate these and pass them into each call.	

		priors: torch.Tensor, shape=(-1, -1, self.n_nodes)
			Prior probabilities of assigning each symbol to each node. If not
			provided, do not include in the calculations (conceptually
			equivalent to a uniform probability, but without scaling the
			probabilities).


		Returns
		-------
		transitions: torch.Tensor, shape=(-1, self.n_nodes, self.n_nodes)
			The expected number of transitions across each edge that occur
			for each example. The returned transitions follow the structure
			of the transition matrix and so will be dense or sparse as
			appropriate.

		responsibility: torch.Tensor, shape=(-1, -1, self.n_nodes)
			The posterior probabilities of each observation belonging to each
			state given that one starts at the beginning of the sequence,
			aligns observations across all paths to get to the current
			observation, and then proceeds to align all remaining observations
			until the end of the sequence.

		starts: torch.Tensor, shape=(-1, self.n_nodes)
			The probabilities of starting at each node given the 
			forward-backward algorithm.

		ends: torch.Tensor, shape=(-1, self.n_nodes)
			The probabilities of ending at each node given the forward-backward
			algorithm.

		logp: torch.Tensor, shape=(-1,)
			The log probabilities of each sequence given the model.
		"""

		n, l, _ = emissions.shape

		f = self.forward(emissions, priors=priors)
		b = self.backward(emissions, priors=priors)

		logp = torch.logsumexp(f[:, -1] + self.ends, dim=1)

		f_ = f[:, :-1].unsqueeze(-1)
		b_ = (b[:, 1:] + emissions[:, 1:]).unsqueeze(-2)

		t = f_ + b_ + self.edges.unsqueeze(0).unsqueeze(0)
		t = t.reshape(n, l-1, -1)
		t = torch.exp(torch.logsumexp(t, dim=1).T - logp).T
		t = t.reshape(n, int(t.shape[1] ** 0.5), -1)

		starts = self.starts + emissions[:, 0] + priors[:, 0] + b[:, 0]
		starts = torch.exp(starts.T - torch.logsumexp(starts, dim=-1)).T

		ends = self.ends + f[:, -1]
		ends = torch.exp(ends.T - torch.logsumexp(ends, dim=-1)).T

		r = f + b
		r = r - torch.logsumexp(r, dim=2).reshape(n, -1, 1)
		return t, r, starts, ends, logp

	def _labeled_summarize(self, X, y):
		"""Extract sufficient statistics given a set of labels.

		This method calculates the sufficient statistics from data where the
		observations have labels. This amounts to essentially counting the
		number of times that each transition occurs and creating a sparse
		update matrix.

		
		Parameters
		----------
		emissions: torch.Tensor, shape=(-1, -1, self.n_nodes)
			Precalculated emission log probabilities. These are the
			probabilities of each observation under each probability 
			distribution. When running some algorithms it is more efficient
			to precalculate these and pass them into each call.	

		y: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, length, self.d)
			A set of labels with the same shape as the observations that
			indicate which node each observation came from. Passing this in
			means that the model uses labeled learning instead of Baum-Welch.
			Default is None.
		"""

		y = _check_parameter(_cast_as_tensor(y), "y", ndim=2, min_value=0, 
			max_value=self.n_nodes-1, dtypes=(torch.int32, torch.int64),
			shape=(X.shape[0], X.shape[1]))

		n, l, d = X.shape

		starts = torch.zeros(n, self.n_nodes, device=self.device)
		starts[torch.arange(n), y[:, 0]] = 1 

		ends = torch.zeros_like(starts)
		ends[torch.arange(n), y[:, -1]] = 1

		t = torch.zeros((n, self.n_nodes, self.n_nodes), device=self.device)
		r = torch.zeros(n, l, self.n_nodes, device=self.device) - inf

		for i in range(n):
			for j in range(l-1):
				t[i][y[i, j], y[i, j+1]] += 1
				r[i, j, y[i, j]] = 0

			r[i, l-1, y[i, l-1]] = 0

		if self._initialized:
			logps = self.log_probability(X)
		else:
			logps = torch.zeros(n, device=self.device)

		return t, r, starts, ends, logps

	def summarize(self, X, y=None, sample_weight=None, emissions=None, 
		priors=None):
		"""Extract the sufficient statistics from a batch of data.

		This method calculates the sufficient statistics from optionally
		weighted data and adds them to the stored cache. The examples must be
		given in a 2D format. Sample weights can either be provided as one
		value per example or as a 2D matrix of weights for each feature in
		each example.


		Parameters
		----------
		X: torch.Tensor, shape=(-1, -1, self.d)
			A set of examples to summarize.

		y: torch.Tensor, shape=(-1, -1), optional 
			A set of labels with the same number of examples and length as the
			observations that indicate which node in the model that each
			observation should be assigned to. Passing this in means that the
			model uses labeled training instead of Baum-Welch. Default is None.

		sample_weight: torch.Tensor, optional
			A set of weights for the examples. This can be either of shape
			(-1, self.d) or a vector of shape (-1,). Default is ones.

		emissions: torch.Tensor, shape=(-1, -1, self.n_nodes)
			Precalculated emission log probabilities. These are the
			probabilities of each observation under each probability 
			distribution. When running some algorithms it is more efficient
			to precalculate these and pass them into each call.	

		priors: torch.Tensor, shape=(-1, -1, self.d)
			Prior probabilities of assigning each symbol to each node. If not
			provided, do not include in the calculations (conceptually
			equivalent to a uniform probability, but without scaling the
			probabilities).
		"""

		if y is None:
			t, r, starts, ends, logps = self.forward_backward(emissions, 
				priors=priors)
		else:
			t, r, starts, ends, logps = self._labeled_summarize(emissions, 
				y=y)

		self._xw_starts_sum += torch.sum(starts * sample_weight, dim=0)
		self._xw_ends_sum += torch.sum(ends * sample_weight, dim=0)
		self._xw_sum += torch.sum(t * sample_weight.unsqueeze(-1), dim=0) 

		X = X.reshape(-1, X.shape[-1])
		r = torch.exp(r) * sample_weight.unsqueeze(-1)
		for i, node in enumerate(self.nodes):
			w = r[:, :, i].reshape(-1, 1)
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

		node_out_count = torch.sum(self._xw_sum, dim=1, keepdims=True)
		node_out_count += self._xw_ends_sum.unsqueeze(1)

		ends = torch.log(self._xw_ends_sum / node_out_count[:,0])
		starts = torch.log(self._xw_starts_sum / self._xw_starts_sum.sum())
		edges = torch.log(self._xw_sum / node_out_count)

		_update_parameter(self.ends, ends, inertia=self.inertia)
		_update_parameter(self.starts, starts, inertia=self.inertia)
		_update_parameter(self.edges, edges, inertia=self.inertia)
		self._reset_cache()
