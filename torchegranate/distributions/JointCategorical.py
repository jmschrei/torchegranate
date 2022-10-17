# JointCategorical.py

import torch

from .multinomial import Multinomial

class JointCategorical(torch.nn.Module):
	def __init__(self, probs=None, n_categories=None, pseudocount=0):
		self.probs = probs
		self.n_categories = n_categories
		self.pseudocount = pseudocount

		if self.n_categories is not None:
			self.n_categories = n_categories
		elif self.probs is not None:
			self.n_categories = probs.shape
		else:
			raise ValueError()

		self._counts = torch.zeros(*self.n_categories)

	def log_probability(self, X):
		log_probabilities = torch.zeros(len(X))
		for i in range(len(X)):
			log_probabilities[i] = self.probs[tuple(X[i])]

		return log_probabilities

	def marginal(self, dims=0):
		dims = tuple(i for i in range(self.probs.ndim) if i != dims)
		return Multinomial(self.probs.sum(dim=dims).unsqueeze(1))

	def summarize(self, X, sample_weights=None):
		if sample_weights is None:
			sample_weights = torch.ones(len(X))

		for i in range(len(X)):
			self._counts[tuple(X[i])] += sample_weights[i]

	def from_summaries(self):
		self._counts += self.pseudocount
		self.probs = self._counts / self._counts.sum()

		self._counts *= 0
