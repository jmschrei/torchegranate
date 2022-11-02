# student_t.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import math
import torch

from .._utils import _cast_as_tensor
from .._utils import _update_parameter
from .._utils import _check_parameter

from .normal import Normal


class StudentT(Normal):
	def __init__(self, dofs, means=None, covs=None, inertia=0.0, frozen=False):
		self.name = "StudentT"

		self.dofs = _check_parameter(_cast_as_tensor(dofs), "dofs", 
			min_value=1, ndim=0, dtypes=(torch.int32, torch.int64))
		self._lgamma_dofsp1 = torch.lgamma((self.dofs + 1) / 2.0)
		self._lgamma_dofs = torch.lgamma(self.dofs / 2.0)

		super().__init__(means=means, covs=covs, covariance_type='diag', 
			inertia=inertia, frozen=frozen)


	def _reset_cache(self):
		super()._reset_cache()
		if self._initialized == False:
			return

		self._log_sqrt_dofs_pi_cov = torch.log(torch.sqrt(self.dofs * math.pi * 
			self.covs))

	def log_probability(self, X):
		X = _check_parameter(_cast_as_tensor(X), "X", ndim=2, 
			shape=(-1, self.d))

		t = (X - self.means) ** 2 / self.covs
		return torch.sum(self._lgamma_dofsp1 - self._lgamma_dofs - \
			self._log_sqrt_dofs_pi_cov -((self.dofs + 1) / 2.0) * 
			torch.log(1 + t / self.dofs), dim=-1)

