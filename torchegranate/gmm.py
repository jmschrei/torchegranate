# gmm.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

import torch
from BayesClassifier import BayesClassifier


class GeneralMixtureModel2(BayesClassifier):
	def __init__(self, distributions=None, priors=None, max_iters=10, threshold=0.1):
		super(GeneralMixtureModel2, self).__init__(distributions=distributions,
			priors=priors)
		self.max_iters = max_iters
		self.threshold = threshold

	def fit(self, X, sample_weights=None):
		initial_logp = self.log_probability(X).sum()
		logp = initial_logp

		max_iters = 20

		for i in range(self.max_iters):
			self.summarize(X, sample_weights=sample_weights)
			self.from_summaries()

			last_logp = logp
			logp = self.log_probability(X).sum()

			print(logp - last_logp)
			if logp - last_logp < 0.1:
				break

	def summarize(self, X, sample_weights=None):
		e = self.predict_proba(X)
		for j, d in enumerate(self.distributions):
			d.summarize(X, e[:, j:j+1])

		self._w_sum += e.sum(dim=0)
		self._n_sum += e.shape[0]



import numpy
import time 

from pomegranate import NormalDistribution
from pomegranate import PoissonDistribution
from pomegranate import IndependentComponentsDistribution
from pomegranate import GeneralMixtureModel

from distributions.normal import Normal
from distributions.poisson import Poisson

d = 100
n = 100000
m = 5

X = numpy.concatenate([numpy.random.randn(n, d) + i / 4. - m // 2 for i in range(m)])
X = numpy.exp(X)

mus = numpy.random.randn(m, d)
mus = numpy.exp(X)

max_iters = 10

tic = time.time()

ds = []
for i in range(m):
	d = [PoissonDistribution(mu) for mu in mus[i]]
	d = IndependentComponentsDistribution(d)
	ds.append(d)

model1 = GeneralMixtureModel(ds, numpy.ones(m) / m)
model1.fit(X, max_iterations=max_iters)
logp1 = model1.log_probability(X)
toc1 = time.time() - tic

X = torch.tensor(X, dtype=torch.float32)
mus = torch.tensor(mus, dtype=torch.float32)
#cov = torch.tensor(cov, dtype=torch.float32)

tic = time.time()
ds = []
for i in range(m):
	d = Poisson(mus[i])
	ds.append(d)

model2 = GeneralMixtureModel2(ds, torch.ones(m) / m, max_iters=max_iters)
model2.fit(X)
logp2 = model2.log_probability(X)
toc2 = time.time() - tic

print(toc1, logp1.sum())
print(toc2, logp2.sum())
print(numpy.abs(logp1 - logp2.numpy()).sum())
