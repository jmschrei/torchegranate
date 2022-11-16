class DenseHiddenMarkovModel(HiddenMarkovModel):
	def __init__(self, distributions, transitions, starts, ends, batch_size=2048, max_iter=10, 
		tol=0.1, inertia=0.0, frozen=False):
		super().__init__(inertia=inertia, frozen=frozen)
		self.name = "HiddenMarkovModel"

		self.distributions = _check_parameter(distributions, "distributions", dtypes=(list, tuple))
		self.nodes = [Node(d, str(i)) for i, d in enumerate(distributions)]
		self.transitions = _check_parameter(_cast_as_tensor(transitions), "transitions")
		self._log_transitions = torch.log(self.transitions)

		self.starts = _check_parameter(_cast_as_tensor(starts), "starts")
		self.ends = _check_parameter(_cast_as_tensor(ends), "ends")

		self.n_nodes = len(self.distributions)

		self._initialized = False
		#self._reset_cache()

		self.expected_transitions = _parameter(torch.zeros(self.n_nodes, self.n_nodes, 
			dtype=torch.float64))


	def forward(self, X, priors=None, emissions=None):
		n, k, d = X.shape
		X, priors, emissions = _check_inputs(self, X, priors, emissions)

		f = torch.zeros(k, n, self.n_nodes, device=self.device, dtype=torch.float64) + float("-inf")
		f[0] = self.starts + emissions[0].T + priors[:, 0]

		
		t_max = self._log_transitions.max()
		t = torch.exp(self._log_transitions - t_max)

		for i in range(1, k):
			p_max = f[i-1].max()
			p = torch.exp(f[i-1] - p_max)
			f[i] = torch.log(torch.matmul(p, t)) + t_max + p_max + emissions[i].T

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
