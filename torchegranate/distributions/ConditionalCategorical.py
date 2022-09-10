# JointCategorical.py

import torch

class ConditionalCategorical(torch.nn.Module):
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
		return Categorical(self.probs.sum(axis=axis))

	def summarize(self, X, sample_weights=None):
		if sample_weights is None:
			sample_weights = torch.ones(len(X))

		for i in range(len(X)):
			self._counts[tuple(X[i])] += sample_weights[i]

	def from_summaries(self):
		self._counts += self.pseudocount
		self.probs = self._counts / self._counts.sum()

		self._counts *= 0


ds = 8, 6, 3, 5, 5, 6, 5, 2, 2, 3, 5
d = len(ds)

probs = torch.randn(*ds)
probs = torch.abs(probs)
probs /= probs.sum(dims=range(d-1), keepdims=True)

n = 100000

import time

Xs = []
for i in range(d):
	X = torch.randint(ds[i], size=(n, 1))
	Xs.append(X)

X = torch.hstack(Xs)

d = JointProbabilityTable(probs)

tic = time.time()
print(d.log_probability(X).sum())

d.summarize(X)
d.from_summaries()

print(d.log_probability(X).sum())

print(time.time() - tic)