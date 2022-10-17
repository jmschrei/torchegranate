# normal.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import torch

from ._utils import _cast_as_tensor
from ._utils import _update_parameter

from ._distribution import Distribution


# Define some useful constants
NEGINF = float("-inf")
INF = float("inf")
SQRT_2_PI = 2.50662827463
LOG_2_PI = 1.83787706641


class Normal(Distribution):

	"""A normal distribution based on a mean and standard deviation."""
	def __init__(self, means=None, covs=None, covariance_type='full', min_cov=0.0, inertia=0.0, frozen=False):
		super().__init__()
		self.name = "Normal"
		self.covariance_type = covariance_type
		self.min_cov = min_cov
		self.inertia = inertia
		self.frozen = frozen

		self.means = _cast_as_tensor(means)
		self.covs = _cast_as_tensor(covs)

		self._initialized = means is not None
		self.d = len(self.means) if self._initialized else None
		self._reset_cache()


	def _initialize(self, d):
		if self.covariance_type == 'full':
			self.means = torch.zeros(d)
			self.covs = torch.zeros(d, d)

			self._column_sum = torch.zeros(d)
			self._column_w_sum = torch.zeros(d)
			self._pair_sum = torch.zeros(d, d)
			self._pair_w_sum = torch.zeros(d, d)

		elif self.covariance_type in ('diag', 'sphere'):
			self.means = torch.zeros(d)
			self.covs = torch.zeros(d)

			self._weight_sum = 0
			self._column_w_sum = torch.zeros(d)
			self._pair_sum = torch.zeros(d)

		self._initialized = True
		super()._initialize(d)

	def _reset_cache(self):
		if self._initialized == False:
			return

		if self.covariance_type == 'full':
			if self.covs.sum() != 0:		
				chol = torch.linalg.cholesky(self.covs, upper=False)
				self._inv_cov = torch.linalg.solve_triangular(chol, torch.eye(len(self.covs)), upper=False).T
				self._inv_cov_dot_mu = torch.matmul(self.means, self._inv_cov)
				self._log_det = -0.5 * torch.linalg.slogdet(self.covs)[1]

		elif self.covariance_type in ('diag', 'sphere'):
			if self.covs.sum() != 0:
				self._log_sigma_sqrt_2_pi = -torch.log(torch.sqrt(self.covs) * SQRT_2_PI)


	def log_probability(self, X):
		X = _cast_as_tensor(X)

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
		if self.frozen == True:
			return

		X, sample_weights = super().summarize(X, sample_weights=sample_weights)

		if self.covariance_type == 'full':
			self._column_sum += torch.sum(X, axis=0)
			self._column_w_sum += torch.sum(X * sample_weights, axis=0)
			self._pair_sum += torch.matmul(X.T, X)
			self._pair_w_sum += torch.matmul(X.T * sample_weights, X)

		elif self.covariance_type in ('diag', 'sphere'):
			self._weight_sum += torch.sum(sample_weights, axis=0)

			print(X.T.shape, sample_weights.shape, self._column_w_sum.shape)

			self._column_w_sum += torch.sum(X * sample_weights, axis=0)
			self._pair_sum += torch.sum(X ** 2, dim=0)

	def from_summaries(self):
		if self.frozen == True:
			return

		if self.covariance_type == 'full':
			means = self._column_sum / self._column_w_sum
			_column_sum = self._column_sum.reshape(-1, 1) * 2
			covs = (self._pair_sum - _column_sum * _column_sum.T / self._pair_w_sum) / self._pair_w_sum

			print(covs)
			print(self._pair_sum)
			print(_column_sum * _column_sum.T)
			print(_column_sum * _column_sum.T / self._pair_w_sum)

		elif self.covariance_type == 'diag':
			means = self._column_w_sum / self._weight_sum
			covs = self._pair_sum / self._weight_sum -\
				self._column_w_sum ** 2.0 / self._weight_sum ** 2.0

		_update_parameter(self.means, means, self.inertia)
		_update_parameter(self.covs, covs, self.inertia)
		self._reset_cache()
