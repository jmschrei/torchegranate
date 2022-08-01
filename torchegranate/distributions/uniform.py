# uniform.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import torch

class Uniform():
	def __init__(self, mins, maxs, frozen=False, inertia=0.0):
		self.d = len(lows)
		self.mins = mins
		self.maxs = maxs
		self._logp = -torch.log(maxs - mins)
		self.frozen = frozen
		self.inertia = 0.0

		self._mins = torch.zeros(self.d) + float("inf")
		self._maxs = torch.zeros(self.d) - float("inf")


	def log_probability(self, X):
		return torch.where((X > self.mins) & (X < self.maxs), self._logps)

	def summarize(self, X, sample_weights=None):
		if sample_weights is not None:
			print("Note: passing in `sample_weights' to a uniform distribution does not affect the learned parameters.")

		self._mins = torch.minimum(self._mins, X.min(axis=0))
		self._maxs = torch.maximum(self._maxs, X.max(axis=0))

	def from_summaries(self):
		if frozen == True:
			return

		self.mins = self._mins
		self.maxs = self._maxs