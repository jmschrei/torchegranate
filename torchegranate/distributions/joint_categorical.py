# joint_categorical.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import torch

from .._utils import _cast_as_tensor
from .._utils import _update_parameter
from .._utils import _check_parameter
from .._utils import _reshape_weights

from ._distribution import Distribution
from .categorical import Categorical


class JointCategorical(Distribution):
	"""Still under development."""
	
	def __init__(self, probs=None, n_categories=None, pseudocount=0, inertia=0.0, frozen=False):
		super().__init__(inertia=inertia, frozen=frozen)
		self.name = "JointCategorical"

		self.probs = _check_parameter(_cast_as_tensor(probs), "probs", 
			min_value=0, max_value=1, value_sum=1)

		self.n_categories = _check_parameter(n_categories, "n_categories", min_value=2)
		self.pseudocount = _check_parameter(pseudocount, "pseudocount")

		self._initialized = probs is not None
		self.d = len(self.probs.shape) if self._initialized else None

		if self._initialized:
			if n_categories is None:
				self.n_categories = tuple(self.probs.shape)
			elif isinstance(n_categories, int):
				self.n_categories = (n_categories for i in range(n_categories))
			else:
				self.n_categories = tuple(n_categories)
		else:
			self.n_categories = None

		self._reset_cache()

	def _initialize(self, d, n_categories):
		self.probs = torch.zeros(*n_categories)

		self.n_categories = n_categories
		self._initialized = True
		super()._initialize(d)

	def _reset_cache(self):
		if self._initialized == False:
			return

		self._w_sum = torch.zeros(self.d, dtype=self.probs.dtype)
		self._xw_sum = torch.zeros(*self.n_categories, dtype=self.probs.dtype)

		self._log_probs = torch.log(self.probs)

	def log_probability(self, X):
		X = _check_parameter(_cast_as_tensor(X), "X", 
			value_set=tuple(range(max(self.n_categories)+1)), ndim=2, 
			shape=(-1, self.d))

		logps = torch.zeros(len(X), dtype=self.probs.dtype)
		for i in range(len(X)):
			logps[i] = self._log_probs[tuple(X[i])]

		return logps

	def marginal(self, dims=0):
		dims = tuple(i for i in range(self.probs.ndim) if i != dims)
		return Categorical(self.probs.sum(dim=dims).unsqueeze(1))

	def summarize(self, X, sample_weight=None):
		if self.frozen == True:
			return

		X = _check_parameter(_cast_as_tensor(X), "X", ndim=2, 
			dtypes=(torch.int32, torch.int64))

		if not self._initialized:
			self._initialize(len(X[0]), torch.max(X, dim=0)[0]+1)

		X = _check_parameter(X, "X", shape=(-1, self.d), 
			value_set=tuple(range(max(self.n_categories)+1)))

		sample_weight = _reshape_weights(X, _cast_as_tensor(sample_weight, 
			dtype=torch.float32))[:,0]

		self._w_sum += torch.sum(sample_weight, dim=0)
		for i in range(len(X)):
			self._xw_sum[tuple(X[i])] += sample_weight[i]

	def from_summaries(self):
		if self.frozen == True:
			return

		probs = self._xw_sum / self._w_sum[0]

		_update_parameter(self.probs, probs, self.inertia)
		self._reset_cache()
