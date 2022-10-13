# zeroinflated.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import torch

from .. import gmm

from _utils import _cast_as_tensor
from _utils import _update_parameter

from _distribution import Distribution


class ZeroInflated(Distribution):
	def __init__(self, distribution, alpha, inertia=0.0, frozen=False):
		self.distribution = distribution
		self.alpha = alpha

		self._mixture = gmm.GeneralMixtureModel()