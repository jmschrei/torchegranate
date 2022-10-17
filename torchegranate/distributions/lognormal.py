# lognormal.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import torch
from .normal import Normal

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
