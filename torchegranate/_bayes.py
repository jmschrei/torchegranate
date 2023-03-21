# _bayes.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>


import torch

from ._utils import _cast_as_tensor
from ._utils import _update_parameter
from ._utils import _check_parameter
from ._utils import _reshape_weights

from .distributions._distribution import Distribution


class BayesMixin(torch.nn.Module):
	def _reset_cache(self):
		"""Reset the internally stored statistics.

		This method is meant to only be called internally. It resets the
		stored statistics used to update the model parameters as well as
		recalculates the cached values meant to speed up log probability
		calculations.
		"""

		if self._initialized == False:
			return

		self.register_buffer("_w_sum", torch.zeros(self.k, device=self.device))
		self.register_buffer("_log_priors", torch.log(self.priors))

	def _emission_matrix(self, X, y=None):
		"""Return the emission/responsibility matrix.

		This method returns the log probability of each example under each
		distribution contained in the model with the log prior probability
		of each component added.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)
			A set of examples to evaluate. 

		y: torch.nn.MaskedTensor or None, optional
			An incomplete set of labels where the mask indicates what labels
			are observed. When provided, semi-supervised learning is performed,
			otherwise normal unsupervised learning is performed. Default is
			None.

	
		Returns
		-------
		e: torch.Tensor, shape=(-1, self.k)
			A set of log probabilities for each example under each distribution.
		"""

		X = _check_parameter(_cast_as_tensor(X), "X", ndim=2, 
			shape=(-1, self.d))

		e = torch.empty(X.shape[0], self.k, device=self.device)
		for i, d in enumerate(self.distributions):
			e[:, i] = d.log_probability(X)

		e += self._log_priors

		if y is not None:
			if y._masked_data[y._masked_mask].min() < 0:
				raise ValueError("y must be between 0 and num components.")
			if y._masked_data[y._masked_mask].max() >= e.shape[1]:
				raise ValueError("y must be between 0 and num components.")
			if X.shape[0] != y.shape[0]:
				raise ValueError("X.shape[0] and y.shape[0] must be the same.")
			if len(y.shape) > 1:
				raise ValueError("y must be a masked vector.")

			idxs = torch.where(y._masked_mask)[0]
			e[idxs] = float("-inf")
			e[idxs, y._masked_data[idxs]] = 0

		return e

	def log_probability(self, X):
		"""Calculate the log probability of each example.

		This method calculates the log probability of each example given the
		parameters of the distribution. The examples must be given in a 2D
		format. For a Bernoulli distribution, each entry in the data must
		be either 0 or 1.

		Note: This differs from some other log probability calculation
		functions, like those in torch.distributions, because it is not
		returning the log probability of each feature independently, but rather
		the total log probability of the entire example.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)
			A set of examples to evaluate.


		Returns
		-------
		logp: torch.Tensor, shape=(-1,)
			The log probability of each example.
		"""

		e = self._emission_matrix(X)
		return torch.logsumexp(e, dim=1)

	def predict(self, X):
		"""Calculate the label assignment for each example.

		This method calculates the label for each example as the most likely
		component after factoring in the prior probability.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)
			A set of examples to summarize.


		Returns
		-------
		y: torch.Tensor, shape=(-1,)
			The predicted label for each example.
		"""

		e = self._emission_matrix(X)
		return torch.argmax(e, dim=1)

	def predict_proba(self, X):
		"""Calculate the posterior probabilities for each example.

		This method calculates the posterior probabilities for each example
		under each component of the model after factoring in the prior 
		probability and normalizing across all the components.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)
			A set of examples to summarize.


		Returns
		-------
		y: torch.Tensor, shape=(-1, self.k)
			The posterior probabilities for each example under each component.
		"""

		e = self._emission_matrix(X)
		return torch.exp(e - torch.logsumexp(e, dim=1, keepdims=True))
		
	def predict_log_proba(self, X):
		"""Calculate the log posterior probabilities for each example.

		This method calculates the log posterior probabilities for each example
		under each component of the model after factoring in the prior 
		probability and normalizing across all the components.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)
			A set of examples to summarize.


		Returns
		-------
		y: torch.Tensor, shape=(-1, self.k)
			The log posterior probabilities for each example under each 
			component.
		"""

		e = self._emission_matrix(X)
		return e - torch.logsumexp(e, dim=1, keepdims=True)

	def from_summaries(self):
		"""Update the model parameters given the extracted statistics.

		This method uses calculated statistics from calls to the `summarize`
		method to update the distribution parameters. Hyperparameters for the
		update are passed in at initialization time.

		Note: Internally, a call to `fit` is just a successive call to the
		`summarize` method followed by the `from_summaries` method.
		"""

		for d in self.distributions:
			d.from_summaries()

		if self.frozen == True:
			return

		priors = self._w_sum / torch.sum(self._w_sum)

		_update_parameter(self.priors, priors, self.inertia)
		self._reset_cache()
