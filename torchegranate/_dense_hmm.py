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


def _convert_to_dense_edges(nodes, edges, starts, ends, start, end):
	n = len(nodes)

	if len(edges[0]) == n:
		edges = torch.tensor(edges, dtype=torch.float64)
		starts = torch.tensor(starts, dtype=torch.float64)
		ends = torch.tensor(ends, dtype=torch.float64)
		return torch.log(edges), torch.log(starts), torch.log(ends)

	else:
		starts = torch.zeros(n, dtype=torch.float64) + NEGINF
		ends = torch.zeros(n, dtype=torch.float64) + NEGINF
		_edges = torch.zeros(n, n, dtype=torch.float64) + NEGINF

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
	def __init__(self, nodes, edges, start, end, starts=None, ends=None, 
		batch_size=2048, max_iter=10, tol=0.1, inertia=0.0, frozen=False):
		super().__init__(inertia=inertia, frozen=frozen)
		self.name = "_DenseHMM"

		self.start = start
		self.end = end

		self.nodes = torch.nn.ModuleList(nodes)
		self.edges, self.starts, self.ends = _convert_to_dense_edges(nodes, 
			edges, starts, ends, self.start, self.end)

		self.n_nodes = len(nodes)
		self.n_edges = len(edges)

		self._reset_cache()

	def _reset_cache(self):
		self._xw_sum = _parameter(torch.zeros(self.n_nodes, self.n_nodes, 
			dtype=torch.float64))

		self._xw_starts_sum = _parameter(torch.zeros(self.n_nodes, 
			dtype=torch.float64))

		self._xw_ends_sum = _parameter(torch.zeros(self.n_nodes, 
			dtype=torch.float64))

	@property
	def device(self):
		return 'cpu'

	def forward(self, X, priors=None, emissions=None):
		X, priors, emissions = _check_hmm_inputs(self, X, priors, emissions)
		n, k, d = X.shape

		f = torch.zeros(k, n, self.n_nodes, device=self.device, dtype=torch.float64) + float("-inf")
		f[0] = self.starts + emissions[0].T + priors[:, 0]

		t_max = self.edges.max()
		t = torch.exp(self.edges - t_max)


		for i in range(1, k):
			p_max = torch.max(f[i-1], dim=1, keepdims=True).values
			p = torch.exp(f[i-1] - p_max)

			f[i] = torch.log(torch.matmul(p, t)) + t_max + p_max + emissions[i].T

		f = f.permute(1, 0, 2)
		return f

	def backward(self, X, priors=None, emissions=None):
		X, priors, emissions = _check_hmm_inputs(self, X, priors, emissions)
		n, k, d = X.shape

		b = torch.zeros(k, n, self.n_nodes, device=self.device, dtype=torch.float64) + float("-inf")
		b[-1] = self.ends

		t_max = self.edges.max()
		t = torch.exp(self.edges.T - t_max)

		for i in range(k-2, -1, -1):
			p = b[i+1] + emissions[i+1].T
			p_max = torch.max(p, dim=1, keepdims=True).values
			p = torch.exp(p - p_max)

			b[i] = torch.log(torch.matmul(p, t)) + t_max + p_max

		b = b.permute(1, 0, 2)
		return b

	def forward_backward(self, X, priors=None, emissions=None):
		X, priors, emissions = _check_hmm_inputs(self, X, priors, None)
		n, k, d = X.shape

		f = self.forward(X, priors=priors, emissions=emissions)
		b = self.backward(X, priors=priors, emissions=emissions)

		logp = torch.logsumexp(f[:, -1] + self.ends, dim=1)

		f_ = f[:, :-1].unsqueeze(-1)
		b_ = (b[:, 1:] + emissions[1:].permute(2, 0, 1)).unsqueeze(-2)

		t = f_ + b_ + self.edges.unsqueeze(0).unsqueeze(0)
		t = t.reshape(n, k-1, -1)
		t = torch.exp(torch.logsumexp(t, dim=1).T - logp).T
		t = t.reshape(n, int(t.shape[1] ** 0.5), -1)

		starts = self.starts + emissions[0].T + priors[:, 0] + b[:, 0]
		starts = torch.exp(starts.T - torch.logsumexp(starts, dim=-1)).T

		ends = self.ends + f[:, -1]
		ends = torch.exp(ends.T - torch.logsumexp(ends, dim=-1)).T

		fb = f + b
		fb = (fb - torch.logsumexp(fb, dim=2).reshape(len(X), -1, 1))
		return t, fb, starts, ends, logp

	def summarize(self, X, sample_weight=None, priors=None):
		transitions, emissions, starts, ends, logps = self.forward_backward(X, 
			priors=priors)

		self._xw_starts_sum += torch.sum(starts * sample_weight, dim=0)
		self._xw_ends_sum += torch.sum(ends * sample_weight, dim=0)
		self._xw_sum += torch.sum(transitions * sample_weight.unsqueeze(-1), 
			dim=0) 

		X = X.reshape(-1, X.shape[-1])
		emissions = torch.exp(emissions) * sample_weight.unsqueeze(-1)
		for i, node in enumerate(self.nodes):
			w = emissions[:, :, i].reshape(-1, 1)
			node.distribution.summarize(X, sample_weight=w)

		return logps

	def from_summaries(self):
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
