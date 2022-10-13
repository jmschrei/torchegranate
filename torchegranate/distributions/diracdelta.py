# diracdelta.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>


import torch

from _utils import _cast_as_tensor
from _utils import _update_parameter

from _distribution import Distribution


class DiracDelta(Distribution):
	def __init__(self, alpha=0.0, inertia=0.0, frozen=False):
		self.alpha = alpha

	def _initialize(self, d):
		self.alpha = 0.0

		self._initialized = True
		super()._initialize(d)

	def _reset_cache(self):
		return

	def log_probability(self, X):
		X = _cast_as_tensor(X)
		return torch.sum(torch.where(X == 0.0, self.alpha, float("-inf")), dim=-1)

	def summarize(self, X, sample_weights=None):
		return

	def from_summaries(self):
		return

import time

n = 100
d = 10

X = torch.zeros(n, d)

tic = time.time()
d2 = DiracDelta()
logp2 = d2.log_probability(X)
toc2 = time.time() - tic

print("torchegranate time: {:4.4}, torchegranate logp: {:4.4}".format(toc2, logp2.sum()))
