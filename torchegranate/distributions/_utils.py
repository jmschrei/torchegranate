# _utils.py
# Jacob Schreiber <jmschreiber91@gmail.com>

import numpy
import torch

def _cast_as_tensor(value):
	"""Set the parameter.""" 

	if value is None:
		return None

	if isinstance(value, (torch.nn.Parameter, torch.Tensor)):
		return value

	if isinstance(value, (float, int, list, numpy.ndarray)):
		return torch.tensor(value)

def _update_parameter(value, new_value, inertia=0.0):
	"""Update a parameters unles.
	"""

	if hasattr(value, "frozen") and getattr(value, "frozen") == True:
		return

	value[:] = inertia*value + (1-inertia)*new_value