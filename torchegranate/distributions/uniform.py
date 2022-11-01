# uniform.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import torch

from .._utils import _cast_as_tensor
from .._utils import _update_parameter
from .._utils import _check_parameter
from .._utils import _check_shapes

from ._distribution import Distribution


class Uniform(Distribution):
	def __init__(self, mins=None, maxs=None, inertia=0.0, frozen=False):
		super().__init__(inertia=inertia, frozen=frozen)
		self.name = "Uniform"

		self.mins = _check_parameter(_cast_as_tensor(mins), "mins", ndim=1)
		self.maxs = _check_parameter(_cast_as_tensor(maxs), "maxs", ndim=1)

		_check_shapes([self.mins, self.maxs], ["mins", "maxs"])

		self._initialized = (mins is not None) and (maxs is not None)
		self.d = len(self.mins) if self._initialized else None
		self._reset_cache()

	def _initialize(self, d):
		self.mins = torch.zeros(d)
		self.maxs = torch.zeros(d)

		self._initialized = True
		super()._initialize(d)

	def _reset_cache(self):
		if self._initialized == False:
			return

		self._x_mins = torch.zeros(self.d) + float("inf")
		self._x_maxs = torch.zeros(self.d) - float("inf")
		self._logps = -torch.log(self.maxs - self.mins)

	def log_probability(self, X):
		X = _check_parameter(_cast_as_tensor(X), "X", ndim=2, 
			shape=(-1, self.d))

		return torch.where((X >= self.mins) & (X < self.maxs), self._logps, 
			float("-inf")).sum(dim=1)

	def summarize(self, X, sample_weight=None):
		if self.frozen == True:
			return

		X, sample_weight = super().summarize(X, sample_weight=sample_weight)

		self._x_mins = torch.minimum(self._x_mins, X.min(dim=0).values)
		self._x_maxs = torch.maximum(self._x_maxs, X.max(dim=0).values)

	def from_summaries(self):
		if self.frozen == True:
			return

		_update_parameter(self.mins, self._x_mins, self.inertia)
		_update_parameter(self.maxs, self._x_maxs, self.inertia)
		self._reset_cache()
