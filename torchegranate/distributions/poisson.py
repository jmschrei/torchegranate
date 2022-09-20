# poisson.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import torch

class Poisson():
	def __init__(self, lambdas, frozen=False, inertia=0.0):
		self.d = len(lambdas)
		self.lambdas = lambdas
		self.log_lambdas = torch.log(lambdas)
		self.name = "Poisson"
		self.frozen = frozen
		self.inertia = inertia

		self._w_sum = torch.zeros(self.d)
		self._xw_sum = torch.zeros(self.d)


	def log_probability(self, X):
		return torch.sum(X * self.log_lambdas - self.lambdas - torch.lgamma(X+1), dim=-1)

	def summarize(self, X, sample_weights=None):
		if sample_weights is None:
			sample_weights = torch.ones(X.shape[0], 1)

		self._w_sum += torch.sum(sample_weights, axis=(0, 1))
		self._xw_sum += torch.sum(X * sample_weights, dim=0)

	def from_summaries(self):
		if self._w_sum.sum() < 1e-8 or self.frozen == True:
			return

		lambdas = self._xw_sum / self._w_sum
		log_lambdas = torch.log(lambdas)

		self.lambdas = lambdas
		self.log_lambdas = log_lambdas
		#self.clear_summaries()

import numpy
import time 

from pomegranate import IndependentComponentsDistribution
from pomegranate import PoissonDistribution

d = 4500
n = 10000

mu = numpy.random.randn(d) * 15
X = numpy.exp(numpy.random.randn(n, d))

tic = time.time()
d1 = IndependentComponentsDistribution.from_samples(X, distributions=PoissonDistribution)
logp1 = d1.log_probability(X)
toc1 = time.time() - tic

muz = torch.tensor([d.parameters[0] for d in d1.distributions])
X = torch.tensor(X, dtype=torch.float32)

tic = time.time()
d2 = Poisson(muz)
d2.summarize(X)
d2.from_summaries()
logp2 = d2.log_probability(X)
toc2 = time.time() - tic

print(toc1, logp1.sum())
print(toc2, logp2.sum())
print(numpy.abs(logp1 - logp2.numpy()).sum())
