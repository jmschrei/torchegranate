# zeroinflated.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import torch

from .._utils import _cast_as_tensor
from .._utils import _update_parameter
from .._utils import _check_parameter

from ._distribution import Distribution

from ..gmm import GeneralMixtureModel

from .dirac_delta import DiracDelta


class ZeroInflated(GeneralMixtureModel):
	def __init__(self, distribution, alpha=1.0, inertia=0.0, frozen=False):
		super(ZeroInflated, self).__init__(
			distributions=[DiracDelta(), distribution], priors=[1.0, alpha])
