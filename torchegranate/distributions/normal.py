# normal.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import torch

# Define some useful constants
NEGINF = float("-inf")
INF = float("inf")
SQRT_2_PI = 2.50662827463
LOG_2_PI = 1.83787706641


class Normal():
	"""A normal distribution based on a mean and standard deviation."""
	def __init__(self, means, covs, covariance_type='full', frozen=False, min_cov=0.0, inertia=0.0):
		self.d = len(means)
		self.means = torch.tensor(means)
		self.covs = torch.tensor(covs)
		self.name = "Normal"
		self.frozen = frozen
		self.summaries = [0, 0, 0]
		self.min_cov = min_cov
		self.inertia = inertia
		self.covariance_type = covariance_type

		if covariance_type == 'full':
			chol = torch.linalg.cholesky(self.covs, upper=False)
			self._inv_cov = torch.linalg.solve_triangular(chol, torch.eye(self.d), upper=False).T
			self._inv_cov_dot_mu = torch.matmul(self.means, self._inv_cov)
			self._log_det = -0.5 * torch.linalg.slogdet(self.covs)[1]

			self._column_sum = torch.zeros(self.d)
			self._column_w_sum = torch.zeros(self.d)
			self._pair_sum = torch.zeros(self.d, self.d)
			self._pair_w_sum = torch.zeros(self.d, self.d)

		elif covariance_type in ('diag', 'sphere'):
			self._log_sigma_sqrt_2_pi = -torch.log(torch.sqrt(self.covs) * SQRT_2_PI)

			self._weight_sum = 0
			self._column_w_sum = torch.zeros(self.d)
			self._pair_sum = torch.zeros(self.d)



	def __reduce__(self):
		"""Serialize distribution for pickling."""
		return self.__class__, (self.mu, self.sigma, self.frozen, self.min_std)

	def log_probability(self, X):
		if self.covariance_type == 'full':
			logp = torch.matmul(X, self._inv_cov) - self._inv_cov_dot_mu
			logp = self.d * LOG_2_PI + torch.sum(logp **2, dim=-1)
			logp = self._log_det - 0.5 * logp
		
		elif self.covariance_type in ('diag', 'sphere'):
			logp = ((X - self.means) ** 2) / self.covs
			logp = self._log_sigma_sqrt_2_pi - 0.5 * logp
			logp = torch.sum(logp, dim=-1)

		return logp


	def summarize(self, X, sample_weights=None):
		if sample_weights is None:
			sample_weights = torch.ones(X.shape[0])


		if self.covariance_type == 'full':
			self._column_sum += torch.sum(X, axis=0)
			self._column_w_sum += torch.sum(X.T * sample_weights, axis=1)
			self._pair_sum += torch.matmul(X.T, X)
			self._pair_w_sum += torch.matmul(X.T * sample_weights, X)

		elif self.covariance_type in ('diag', 'sphere'):
			self._weight_sum += torch.sum(sample_weights, axis=0)
			self._column_w_sum += torch.sum(X.T * sample_weights, axis=1)
			self._pair_sum += torch.sum(X ** 2, dim=0)

	def from_summaries(self):
		if self._weight_sum < 1e-8 or self.frozen == True:
			return

		if self.covariance_type == 'full':
			pass

		elif self.covariance_type == 'diag':
			means = self._column_w_sum / self._weight_sum
			covs = self._pair_sum / self._weight_sum -\
				self._column_w_sum ** 2.0 / self._weight_sum ** 2.0



		self.means = self.means*self.inertia + means*(1.-self.inertia)
		self.covs = self.covs*self.inertia + covs*(1.-self.inertia)
		#self.covs = torch.maximum(self.covs, self.min_cov)

		if self.covariance_type == 'diag':
			 self._log_sigma_sqrt_2_pi = -torch.log(torch.sqrt(self.covs) * SQRT_2_PI)


import numpy
import time 

from pomegranate import MultivariateGaussianDistribution
from pomegranate import IndependentComponentsDistribution
from pomegranate import NormalDistribution

d = 1500
n = 100000

z = 3.7784

mu = numpy.random.randn(d) * 15
cov = numpy.eye(d) * z #* numpy.abs(numpy.random.randn(d))


X = numpy.random.randn(n, d)

tic = time.time()
d1 = IndependentComponentsDistribution.from_samples(X, distributions=NormalDistribution)
logp1 = d1.log_probability(X)
toc1 = time.time() - tic

X = torch.tensor(X, dtype=torch.float32)

tic = time.time()
d2 = Normal(mu, z, covariance_type='diag')
d2.summarize(X[:100])
d2.summarize(X[100:5000])
d2.summarize(X[5000:])
d2.from_summaries()
logp2 = d2.log_probability(X)
toc2 = time.time() - tic

print(toc1, logp1.sum())
print(toc2, logp2.sum())
print(numpy.abs(logp1 - logp2.numpy()).sum())