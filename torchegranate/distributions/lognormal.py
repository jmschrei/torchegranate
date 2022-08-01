# lognormal.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import torch
from normal import Normal

class LogNormal(Normal):
	def __init__(self, means, covs, covariance_type='diag', min_std=0.0, frozen=False, inertia=0.0):
		super(LogNormal, self).__init__(means=means, covs=covs, 
			covariance_type=covariance_type, min_std=min_std, 
			frozen=frozen, inertia=inertia)

	def log_probability(self, X):
		log_X = torch.log(X)
		return super(LogNormal, self).log_probability(X=log_X)

	def summarize(self, X, sample_weights=None):
		log_X = torch.log(X)
		super(LogNormal, self).summarize(X=log_X, sample_weights=sample_weights)

	def from_summaries(self):
		super(LogNormal, self).from_summaries()

import numpy
import time 

from pomegranate import MultivariateGaussianDistribution
from pomegranate import IndependentComponentsDistribution
from pomegranate import LogNormalDistribution

d = 150
n = 1000

z = 3.7784

mu = numpy.random.randn(d) * 15
cov = numpy.eye(d) * z #* numpy.abs(numpy.random.randn(d))


X = numpy.exp(numpy.random.randn(n, d))

tic = time.time()
d1 = IndependentComponentsDistribution.from_samples(X, distributions=LogNormalDistribution)
logp1 = d1.log_probability(X)
toc1 = time.time() - tic

X = torch.tensor(X, dtype=torch.float32)

tic = time.time()
d2 = LogNormal(mu, z, covariance_type='diag')
#d2.summarize(X)
#d2.from_summaries()
logp2 = d2.log_probability(X)
toc2 = time.time() - tic

print(toc1, logp1.sum())
print(toc2, logp2.sum())
print(numpy.abs(logp1 - logp2.numpy()).sum())