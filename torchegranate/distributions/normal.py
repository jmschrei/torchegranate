# normal.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import torch

from .._utils import _cast_as_tensor
from .._utils import _update_parameter
from .._utils import _check_parameter
from .._utils import _check_shapes

from ._distribution import Distribution


# Define some useful constants
NEGINF = float("-inf")
INF = float("inf")
SQRT_2_PI = 2.50662827463
LOG_2_PI = 1.83787706641


class Normal(Distribution):

	"""A normal distribution based on a mean and standard deviation."""
	def __init__(self, means=None, covs=None, covariance_type='full', 
		min_cov=0.0, inertia=0.0, frozen=False):
		super().__init__(inertia=inertia, frozen=frozen)
		self.name = "Normal"

		ndim = 1 if covariance_type in ('diag', 'sphere') else 2

		self.means = _check_parameter(_cast_as_tensor(means), "means", ndim=1)
		self.covs = _check_parameter(_cast_as_tensor(covs), "covs", ndim=ndim)

		_check_shapes([self.means, self.covs], ["means", "covs"])

		self.min_cov = _check_parameter(min_cov, "min_cov", min_value=0, ndim=0)
		self.covariance_type = covariance_type

		self._initialized = (means is not None) and (covs is not None)
		self.d = len(means) if self._initialized else None
		self._reset_cache()

	def _initialize(self, d):
		self.means = torch.zeros(d)
		
		if self.covariance_type == 'full':
			self.covs = torch.zeros(d, d)
		elif self.covariance_type == 'diag':
			self.covs = torch.zeros(d)
		elif self.covariance_type == 'sphere':
			self.covs = torch.tensor(0)

		self._initialized = True
		super()._initialize(d)

	def _reset_cache(self):
		if self._initialized == False:
			return

		self._w_sum = torch.zeros(self.d)
		self._xw_sum = torch.zeros(self.d)

		if self.covariance_type == 'full':
			self._xxw_sum = torch.zeros(self.d, self.d)
	
			if self.covs.sum() > 0.0:
				chol = torch.linalg.cholesky(self.covs, upper=False)
				self._inv_cov = torch.linalg.solve_triangular(chol, torch.eye(
					len(self.covs)), upper=False).T
				self._inv_cov_dot_mu = torch.matmul(self.means, self._inv_cov)
				self._log_det = -0.5 * torch.linalg.slogdet(self.covs)[1]
				self._theta = self._log_det - 0.5 * (self.d * LOG_2_PI)

		elif self.covariance_type in ('diag', 'sphere'):
			self._xxw_sum = torch.zeros(self.d)

			if self.covs.sum() > 0.0:
				self._log_sigma_sqrt_2pi = -torch.log(torch.sqrt(self.covs) * 
					SQRT_2_PI)
				self._inv_two_sigma = 1. / (2 * self.covs)

	def log_probability(self, X):
		X = _check_parameter(_cast_as_tensor(X, dtype=self.means.dtype), "X", 
			ndim=2, shape=(-1, self.d))

		if self.covariance_type == 'full':
			logp = torch.matmul(X, self._inv_cov) - self._inv_cov_dot_mu
			logp = self.d * LOG_2_PI + torch.sum(logp ** 2, dim=-1)
			logp = self._log_det - 0.5 * logp
			return logp
		
		elif self.covariance_type in ('diag', 'sphere'):
			return torch.sum(self._log_sigma_sqrt_2pi - ((X - self.means) ** 2) 
				* self._inv_two_sigma, dim=-1)

	def summarize(self, X, sample_weight=None):
		if self.frozen == True:
			return

		X, sample_weight = super().summarize(X, sample_weight=sample_weight)
		X = _cast_as_tensor(X, dtype=self.means.dtype)

		if self.covariance_type == 'full':
			self._w_sum += torch.sum(sample_weight, dim=0)
			self._xw_sum += torch.sum(X * sample_weight, axis=0)
			self._xxw_sum += torch.matmul((X * sample_weight).T, X)

		elif self.covariance_type in ('diag', 'sphere'):
			self._w_sum += torch.sum(sample_weight, dim=0)
			self._xw_sum += torch.sum(X * sample_weight, dim=0)
			self._xxw_sum += torch.sum(X ** 2 * sample_weight, dim=0)

	def from_summaries(self):
		if self.frozen == True:
			return

		means = self._xw_sum / self._w_sum

		if self.covariance_type == 'full':
			v = self._xw_sum.unsqueeze(0) * self._xw_sum.unsqueeze(1)
			covs = self._xxw_sum / self._w_sum -  v / self._w_sum ** 2.0

		elif self.covariance_type == 'diag':
			covs = self._xxw_sum / self._w_sum - \
				self._xw_sum ** 2.0 / self._w_sum ** 2.0

		_update_parameter(self.means, means, self.inertia)
		_update_parameter(self.covs, covs, self.inertia)
		self._reset_cache()
