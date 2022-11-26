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
	def __init__(self, nodes, edges, start, end, starts=None, ends=None, 
		batch_size=2048, max_iter=10, tol=0.1, inertia=0.0, frozen=False):
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


	@property
	def device(self):
		return next(self.parameters()).device

	def _reset_cache(self):
		self._xw_sum = _parameter(torch.zeros(self.n_edges, 
			dtype=torch.float64))

		self._xw_starts_sum = _parameter(torch.zeros(self.n_nodes, 
			dtype=torch.float64))

		self._xw_ends_sum = _parameter(torch.zeros(self.n_nodes, 
			dtype=torch.float64))

	def forward(self, X, priors=None, emissions=None):
		X, priors, emissions = _check_hmm_inputs(self, X, priors, emissions)
		n, k, d = X.shape

		f = torch.zeros(k, n, self.n_nodes, device=self.device, dtype=torch.float64) + float("-inf")
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
		X, priors, emissions = _check_hmm_inputs(self, X, priors, emissions)
		n, k, d = X.shape

		b = torch.zeros(k, n, self.n_nodes, device=self.device, dtype=torch.float64) + float("-inf")
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