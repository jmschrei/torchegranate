# hmm.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy
import torch

from ._utils import _cast_as_tensor
from ._utils import _update_parameter
from ._utils import _check_parameter

from .distributions._distribution import Distribution

from ._bayes import BayesMixin
from ._base import GraphMixin
from ._base import Node

NEGINF = float("-inf")

_parameter = lambda x: torch.nn.Parameter(x, requires_grad=False)


def _check_inputs(model, X, priors, emissions):
	n, k, d = X.shape
	if X.device != model.device:
		X = X.to(model.device)

	if priors is None:
		priors = torch.zeros(1, device=model.device).expand(n, k, model.n_nodes)
	elif priors.device != model.device:
		priors = priors.to(model.device)

	if emissions is None:
		emissions = torch.empty((k, model.n_nodes, n), device=model.device, 
			dtype=torch.float64)
		for i, node in enumerate(model.nodes):
			emissions[:, i] = node.distribution.log_probability(X.reshape(
				-1, d)).reshape(n, k).T

	return X, priors, emissions


class HiddenMarkovModel(GraphMixin, Distribution):
	def __init__(self, nodes=[], edges=[], batch_size=2048, max_iter=10, 
		tol=0.1, inertia=0.0, frozen=False):
		super().__init__(inertia=inertia, frozen=frozen)
		self.name = "HiddenMarkovModel"

		self.nodes = _check_parameter(nodes, "nodes", dtypes=(list, tuple))
		self.edges = _check_parameter(edges, "edges", dtypes=(list, tuple, 
			numpy.array, torch.Tensor))

		self.n_nodes = len(nodes)
		self.n_edges = len(edges)

		self.start = Node(None, "start")
		self.end = Node(None, "end")

		self._initialized = False
		#self._reset_cache()

	@property
	def device(self):
		return next(self.parameters()).device

	def bake(self):
		self.n_nodes = len(self.nodes)
		self.n_edges = len(self.edges)

		self.starts = _parameter(torch.full((self.n_nodes,), NEGINF))
		self.ends = _parameter(torch.full((self.n_nodes,), NEGINF)) 

		self.edge_starts = torch.empty(self.n_edges, dtype=torch.int64)
		self.edge_ends = torch.empty(self.n_edges, dtype=torch.int64)
		self.edge_log_probabilities = torch.empty(self.n_edges, dtype=torch.float64)

		idx = 0
		for x, y, p in self.edges:
			if x is self.start:
				j = self.nodes.index(y)
				self.starts[j] = p

			elif y is self.end:
				i = self.nodes.index(x)
				self.ends[i] = p

			else:
				i = self.nodes.index(x)
				j = self.nodes.index(y)

				self.edge_starts[idx] = i
				self.edge_ends[idx] = j
				self.edge_log_probabilities[idx] = p
				idx += 1

		self.nodes = torch.nn.ModuleList(self.nodes)

		self.edge_starts = _parameter(self.edge_starts[:idx])
		self.edge_ends = _parameter(self.edge_ends[:idx])
		self.edge_log_probabilities = _parameter(self.edge_log_probabilities[:idx])
		self.n_edges = idx

		self.expected_transitions = _parameter(torch.zeros(self.n_edges, 
			dtype=torch.float64))

		self.expected_starts = _parameter(torch.zeros(self.n_nodes, 
			dtype=torch.float64))

		self.expected_ends = _parameter(torch.zeros(self.n_nodes, 
			dtype=torch.float64))


	def forward(self, X, priors=None, emissions=None):
		n, k, d = X.shape
		X, priors, emissions = _check_inputs(self, X, priors, emissions)

		f = torch.zeros(k, n, self.n_nodes, device=self.device, dtype=torch.float64) + float("-inf")
		f[0] = self.starts + emissions[0].T + priors[:, 0]

		for i in range(1, k):
			p = f[i-1, :, self.edge_starts]
			p += self.edge_log_probabilities.expand(n, -1)

			alpha = p.max()
			p = torch.exp(p - alpha)

			z = torch.zeros_like(f[i])
			z.scatter_add_(1, self.edge_ends.expand(n, -1), p)

			f[i] = alpha + torch.log(z) + emissions[i].T + priors[:, i]

		f = f.permute(1, 0, 2)
		return f

	def backward(self, X, priors=None, emissions=None):
		n, k, d = X.shape
		X, priors, emissions = _check_inputs(self, X, priors, emissions)

		b = torch.zeros(k, n, self.n_nodes, device=self.device, dtype=torch.float64) + float("-inf")
		b[-1] = self.ends

		for i in range(k-2, -1, -1):
			p = b[i+1, :, self.edge_ends]
			p += emissions[i+1, self.edge_ends].T + priors[:, i+1, self.edge_ends]
			p += self.edge_log_probabilities.expand(n, -1)

			alpha = p.max()
			p = torch.exp(p - alpha)

			z = torch.zeros_like(b[i])
			z.scatter_add_(1, self.edge_starts.expand(n, -1), p)
			b[i] = alpha + torch.log(z)

		b = b.permute(1, 0, 2)
		return b

	def forward_backward(self, X, priors=None, return_logps=False):
		n, k, d = X.shape
		X, priors, emissions = _check_inputs(self, X, priors, None)

		f = self.forward(X, priors=priors, emissions=emissions)
		b = self.backward(X, priors=priors, emissions=emissions)

		total_logp = torch.logsumexp(f[:, -1] + self.ends, dim=1)

		t = f[:, :-1, self.edge_starts] + b[:, 1:, self.edge_ends]
		t += emissions[1:, self.edge_ends].permute(2, 0, 1) + priors[:, 1:, self.edge_ends]
		t += self.edge_log_probabilities.expand(n, k-1, -1)

		starts = self.starts + emissions[0].T + priors[:, 0] + b[:, 0]
		starts = torch.exp(starts.T - torch.logsumexp(starts, dim=-1)).T

		ends = self.ends + f[:, -1]
		ends = torch.exp(ends.T - torch.logsumexp(ends, dim=-1)).T

		expected_transitions = torch.exp(torch.logsumexp(t, dim=1).T - total_logp).T

		fb = f + b
		fb = (fb - torch.logsumexp(fb, dim=2).reshape(len(X), -1, 1))
		return expected_transitions, fb, starts, ends, total_logp

	def log_probability(self, X, priors=None):
		f = self.forward(X, priors=priors)
		return torch.logsumexp(f[:, -1] + self.ends, dim=1)

	def fit(self, X, priors=None):
		last_logp = None

		for i in range(max_iterations+1):
			start_time = time.time()
			if i > 0:
				self.from_summaries()
			elif i == max_iterations:
				break
			
			logp = 0
			for start in range(0, len(X), batch_size):
				end = start + batch_size
				logp_ = self.summarize(X[start:end], priors=priors[start:end] if priors is not None else None)
				logp += logp_.sum()

			duration = time.time() - start_time

			if last_logp is not None:
				improvement = logp - last_logp
				print("[{}] Improvement: {}, Time: {:4.4}s".format(i, improvement, duration))

			last_logp = logp

	def summarize(self, X, priors=None):
		expected_transitions, fb, starts, ends, logps = self.forward_backward(X, priors=priors)

		self.expected_starts += torch.sum(starts, dim=0)
		self.expected_ends += torch.sum(ends, dim=0)
		self.expected_transitions += torch.sum(expected_transitions, dim=0) 

		fb = torch.exp(fb)
		for i, node in enumerate(self.nodes):
			node.distribution.summarize(X, weights=fb[:,:,i])

		return logps

	def from_summaries(self):
		node_out_count = torch.clone(self.expected_ends)
		for start, count in zip(self.edge_starts, self.expected_transitions):
			node_out_count[start] += count

		self.ends[:] = torch.log(self.expected_starts / node_out_count)
		self.starts[:] = torch.log(self.expected_starts / self.expected_starts.sum())

		for i in range(self.n_edges):
			t = self.expected_transitions[i]
			t_sum = node_out_count[self.edge_starts[i]]
			self.edge_log_probabilities[i] = torch.log(t / t_sum)

		for node in self.nodes:
			node.distribution.from_summaries()

		self.expected_transitions *= 0
		self.expected_starts *= 0
		self.expected_ends *= 0

	@classmethod
	def from_matrix(cls, distributions, transitions, starts, ends, **kwargs):
		nodes = []
		for i, d in enumerate(distributions):
			node = Node(d, name=str(i))
			nodes.append(node)

		edges = []
		for i in range(transitions.shape[0]):
			for j in range(transitions.shape[1]):
				if transitions[i, j] != 0:
					edge = nodes[i], nodes[j], transitions[i, j]
					edges.append(edge)

		model = cls(nodes=nodes, edges=edges, **kwargs)

		for i in range(len(starts)):
			if starts[i] != 0:
				model.add_edge(model.start, nodes[i], starts[i])

		for i in range(len(ends)):
			if ends[i] != 0:
				model.add_edge(nodes[i], model.end, ends[i])

		model.bake() 
		return model

