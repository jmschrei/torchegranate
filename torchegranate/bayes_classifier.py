# BayesClassifier.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

import torch

class BayesClassifier(torch.nn.Module):
	def __init__(self, distributions=None, priors=None, max_iters=10):
		self.distributions = distributions
		self.priors = torch.tensor(priors)
		self.max_iters = max_iters
		self.m = len(distributions)
		self._reset_cache()

	def _initialize(self, m, d):
		self.m = m
		self.priors = torch.ones(m) / m
		
	def _reset_cache(self):
		self._log_priors = torch.log(self.priors)
		self._w_sum = torch.zeros_like(self._log_priors)
		self._n_sum = 0

	def _emission_matrix(self, X):
		n, d = X.shape
		m = len(self.distributions)

		e = torch.empty(n, m, dtype=X.dtype)
		for i, d in enumerate(self.distributions):
			e[:,i] = d.log_probability(X)

		return e + self._log_priors

	def log_probability(self, X):
		e = self._emission_matrix(X)
		return torch.logsumexp(e, dim=1)

	def predict(self, X):
		e = self._emission_matrix(X)
		return torch.argmax(e, dim=1)

	def predict_proba(self, X):
		e = self._emission_matrix(X)
		return torch.exp(e - torch.logsumexp(e, dim=1, keepdims=True))
		
	def predict_log_proba(self, X):
		e = self._emission_matrix(X)
		return e - torch.logsumexp(e, dim=1, keepdims=True)

	def fit(self, X, y, sample_weights=None):
		initial_logp = self.log_probability(X).sum()
		
		self.summarize(X, sample_weights=sample_weights)
		self.from_summaries()

		last_logp = logp
		logp = self.log_probability(X).sum()

	def summarize(self, X, y, sample_weights=None):
		for j, d in enumerate(self.distributions):
			idxs = y == j
			d.summarize(X[idxs], e[idxs, j:j+1])
			self._w_sum[j] += idxs.sum()

		self._n_sum += e.shape[0]

	def from_summaries(self):
		for d in self.distributions:
			d.from_summaries()

		self.priors = self._w_sum / self._n_sum
		self._reset_cache()