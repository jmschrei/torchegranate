# JointCategorical.py

import torch

from categorical import Categorical

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
		return Categorical(self.probs.sum(dim=dims).unsqueeze(1))

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
probs /= probs.sum()

n = 1000

import time

Xs = []
for i in range(d):
	X = torch.randint(ds[i], size=(n, 1))
	Xs.append(X)

X = torch.hstack(Xs)

d = JointCategorical(probs)

tic = time.time()
print(d.log_probability(X).sum())

d.summarize(X)
d.from_summaries()

print(d.log_probability(X).sum())

print(time.time() - tic)


from pomegranate import JointProbabilityTable
from pomegranate import DiscreteDistribution

p1 = DiscreteDistribution({0: 0.2, 1: 0.8})
p2 = DiscreteDistribution({0: 0.6, 1: 0.4})
p3 = DiscreteDistribution({0: 0.1, 1: 0.7, 2: 0.2})

d = JointProbabilityTable([
	[0, 0, 0, 0.05],
	[0, 0, 1, 0.07],
	[0, 0, 2, 0.03],
	[0, 1, 0, 0.02],
	[0, 1, 1, 0.01],
	[0, 1, 2, 0.05],
	[1, 0, 0, 0.02],
	[1, 0, 1, 0.3],
	[1, 0, 2, 0.05],
	[1, 1, 0, 0.1],
	[1, 1, 1, 0.2],
	[1, 1, 2, 0.1]], parents=[p1, p2, p3])

print(d.marginal(2))

p = torch.tensor([[[0.05, 0.07, 0.03],
				   [0.02, 0.01, 0.05]],
				  [[0.02, 0.30, 0.05],
				   [0.10, 0.20, 0.10]]])


a = JointCategorical(p)
print(torch.exp(a.marginal(2).log_probabilities))