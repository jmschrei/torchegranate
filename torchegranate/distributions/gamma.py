# gamma.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import torch

from .._utils import _cast_as_tensor
from .._utils import _update_parameter
from .._utils import _check_parameter
from .._utils import _check_shapes

from ._distribution import Distribution


class Gamma(Distribution):
	"""A gamma distribution object.

	A gamma distribution is the sum of exponential distributions, and has shape
	and rate parameters. This distribution assumes that each feature is
	independent of the others. 

	There are two ways to initialize this objecct. The first is to pass in
	the tensor of rate and shae parameters, at which point they can immediately 
	be used. The second is to not pass in the rate parameters and then call
	either `fit` or `summary` + `from_summaries`, at which point the rate
	and shape parameters will be learned from data.


	Parameters
	----------
	shapes: torch.tensor or None, shape=(d,), optional
		The shape parameter for each feature. Default is None

	rates: torch.tensor or None, shape=(d,), optional
		The rate parameters for each feature. Default is None.

	inertia: float, (0, 1), optional
		Indicates the proportion of the update to apply to the parameters
		during training. When the inertia is 0.0, the update is applied in
		its entirety and the previous parameters are ignored. When the
		inertia is 1.0, the update is entirely ignored and the previous
		parameters are kept, equivalently to if the parameters were frozen.

	tol: float, [0, inf), optional
		The threshold at which to stop fitting the parameters of the
		distribution. Default is 1e-4.

	max_iter: int, [0, inf), optional
		The maximum number of iterations to run EM when fitting the parameters
		of the distribution. Default is 20.

	frozen: bool, optional
		Whether all the parameters associated with this distribution are frozen.
		If you want to freeze individual pameters, or individual values in those
		parameters, you must modify the `frozen` attribute of the tensor or
		parameter directly. Default is False.


	Examples
	--------
	>>> # Create a distribution with known parameters
	>>> shapes = torch.tensor([0.2, 0.6])
	>>> rates = torch.tensor([1.2, 0.4])
	>>> X = torch.tensor([[0.3, 0.2], [0.8, 0.1]])
	>>>
	>>> d = Gamma(shapes, rates)
	>>> d.log_probability(X)
	tensor([-1.2687, -2.3361])
	>>>
	>>>
	>>> # Fit a distribution to data
	>>> n, d = 100, 10
	>>> X = torch.exp(torch.randn(d) * 15)
	>>> 
	>>> d = Gamma().fit(X)
	>>>
	>>>
	>>> # Fit a distribution using the summarize API
	>>> n, d = 100, 10
	>>> X = torch.exp(torch.randn(d) * 15)
	>>> 
	>>> d = Gamma()
	>>> d.summarize(X[:50])
	>>> d.summarize(X[50:])
	>>> d.from_summaries()
	>>>
	>>>
	>>> # As a loss function for a neural network
	>>> class ToyNet(torch.nn.Module):
	>>> 	def __init__(self, d):
	>>>			super(ToyNet, self).__init__()
	>>>			self.fc1 = torch.nn.Linear(d, 32)
	>>>			self.shapes = torch.nn.Linear(32, d)
	>>>			self.rates = torch.nn.Linear(32, d)
	>>>			self.relu = torch.nn.ReLU()
	>>>
	>>>		def forward(self, X):
	>>>			X = self.fc1(X)
	>>>			X = self.relu(X)
	>>>			shapes = self.shapes(X)
	>>>			rates = self.rates(X)
	>>>			return self.relu(shapes) + 0.01, self.relu(rates) + 0.01
	>>>
	>>> n, d = 1000
	>>> X = torch.exp(torch.randn(n, d) * 15)
	>>>
	>>> model = ToyNet(d)
	>>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
	>>>
	>>> for i in range(100):
	>>>		optimizer.zero_grad()
	>>>
	>>>		shapes, rates = model(X)
	>>> 	loss = -Gamma(shapes, 2).log_probability(X).sum()
	>>>		loss.backward()
	>>>		optimizer.step()
	"""

	def __init__(self, shapes=None, rates=None, inertia=0.0, tol=1e-4, 
		max_iter=20, frozen=False):
		super().__init__(inertia=inertia, frozen=frozen)
		self.name = "Gamma"

		self.shapes = _check_parameter(_cast_as_tensor(shapes), "shapes", 
			min_value=0, ndim=1)
		self.rates = _check_parameter(_cast_as_tensor(rates), "rates", 
			min_value=0, ndim=1)

		_check_shapes([self.shapes, self.rates], ["shapes", "rates"])

		self.tol = _check_parameter(tol, "tol", min_value=0, ndim=0)
		self.max_iter = _check_parameter(max_iter, "max_iter", min_value=1,
			ndim=0)

		self._initialized = (shapes is not None) and (rates is not None)
		self.d = len(self.shapes) if self._initialized else None
		self._reset_cache()

	def _initialize(self, d):
		"""Initialize the probability distribution.

		This method ie meant to only be called internally. It initializes the
		parameters of the distribution and stores its dimensionality. For more
		complex methods, this function will do more.


		Parameters
		----------
		d: int
			The dimensionality the distribution is being initialized to.
		"""

		self.shapes = torch.zeros(d)
		self.rates = torch.zeros(d)

		self._initialized = True
		super()._initialize(d)

	def _reset_cache(self):
		"""Reset the internally stored statistics.

		This method is meant to only be called internally. It resets the
		stored statistics used to update the model parameters as well as
		recalculates the cached values meant to speed up log probability
		calculations.
		"""

		if self._initialized == False:
			return

		self._w_sum = torch.zeros(self.d)
		self._xw_sum = torch.zeros(self.d)
		self._logx_w_sum = torch.zeros(self.d)

		self._log_rates = torch.log(self.rates)
		self._lgamma_shapes = torch.lgamma(self.shapes)
		self._thetas = self._log_rates * self.shapes - self._lgamma_shapes

	def log_probability(self, X):
		"""Calculate the log probability of each example.

		This method calculates the log probability of each example given the
		parameters of the distribution. The examples must be given in a 2D
		format. For a gamma distribution, the data must be non-negative.

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

		X = _check_parameter(_cast_as_tensor(X), "X", min_value=0.0, 
			ndim=2, shape=(-1, self.d))

		return torch.sum(self._thetas + torch.log(X) * (self.shapes - 1) - 
			self.rates * X, dim=-1)

	def summarize(self, X, sample_weight=None):
		"""Extract the sufficient statistics from a batch of data.

		This method calculates the sufficient statistics from optionally
		weighted data and adds them to the stored cache. The examples must be
		given in a 2D format. Sample weights can either be provided as one
		value per example or as a 2D matrix of weights for each feature in
		each example.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)
			A set of examples to summarize.

		sample_weight: list, tuple, numpy.ndarray, torch.Tensor, optional
			A set of weights for the examples. This can be either of shape
			(-1, self.d) or a vector of shape (-1,). Default is ones.
		"""

		if self.frozen == True:
			return

		X, sample_weight = super().summarize(X, sample_weight=sample_weight)
		X = _check_parameter(X, "X", min_value=0)

		self._w_sum += torch.sum(sample_weight, dim=0)
		self._xw_sum += torch.sum(X * sample_weight, dim=0)
		self._logx_w_sum += torch.sum(torch.log(X) * sample_weight, dim=0)

	def from_summaries(self):
		"""Update the model parameters given the extracted statistics.

		This method uses calculated statistics from calls to the `summarize`
		method to update the distribution parameters. Hyperparameters for the
		update are passed in at initialization time.

		Note: Internally, a call to `fit` is just a successive call to the
		`summarize` method followed by the `from_summaries` method.
		"""
		
		if self.frozen == True:
			return

		thetas = torch.log(self._xw_sum / self._w_sum) - \
			self._logx_w_sum / self._w_sum

		numerator = (3 - thetas + torch.sqrt((thetas - 3) ** 2 + 24 * thetas))
		denominator = (12 * thetas)

		new_shapes = numerator / denominator
		shapes = new_shapes + self.tol

		for iteration in range(self.max_iter):
			mask = torch.abs(shapes - new_shapes) < self.tol
			if torch.all(mask):
				break

			shapes = new_shapes
			new_shapes = (shapes - (torch.log(shapes) - torch.polygamma(0, 
				shapes) - thetas) / (1.0 / shapes - torch.polygamma(1, shapes)))

		shapes = new_shapes
		rates = 1.0 / (1.0 / (shapes * self._w_sum) * self._xw_sum)

		_update_parameter(self.shapes, shapes, self.inertia)
		_update_parameter(self.rates, rates, self.inertia)
		self._reset_cache()
