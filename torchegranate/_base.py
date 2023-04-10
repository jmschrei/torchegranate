# _base.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

import math
import time
import numpy
import torch

from ._utils import _cast_as_tensor
from ._utils import _update_parameter
from ._utils import _check_parameter
from ._utils import _reshape_weights

from .distributions._distribution import Distribution

from .kmeans import KMeans

NEGINF = float("-inf")
_parameter = lambda x: torch.nn.Parameter(x, requires_grad=False)


def _check_inputs(model, X, emissions, priors):
	if X is None and emissions is None:
		raise ValueError("Must pass in one of `X` or `emissions`.")

	X = _check_parameter(_cast_as_tensor(X), "X", ndim=3, 
		shape=(-1, -1, model.d))
	n = X.shape[0] if X is not None else -1
	k = X.shape[1] if X is not None else -1

	emissions = _check_parameter(_cast_as_tensor(emissions), "emissions", 
		ndim=3, shape=(n, k, model.n_distributions))
	if emissions is None:
		emissions = model._emission_matrix(X)

	priors = _check_parameter(_cast_as_tensor(priors), "priors", ndim=3,
		shape=(n, k, model.n_distributions))
	if priors is None:
		priors = torch.zeros(1, dtype=emissions.dtype, 
			device=model.device).expand_as(emissions)

	return emissions, priors


class Silent(torch.nn.Module):
	def __init__(self):
		super().__init__()


class GraphMixin(torch.nn.Module):
	def add_node(self, node):
		self.nodes.append(node)

	def add_nodes(self, nodes):
		self.nodes.extend(nodes)

	def add_edge(self, start, end, probability=None):
		if probability is None:
			edge = start, end
		else:
			edge = start, end, probability

		self.edges.append(edge)


class _BaseHMM(Distribution):
	"""A base hidden Markov model.

	A hidden Markov model is an extension of a mixture model to sequences by
	including a transition matrix between the elements of the mixture. Each of
	the algorithms for a hidden Markov model are essentially just a revision
	of those algorithms to incorporate this transition matrix.

	There are two main ways one can implement a hidden Markov model: either the
	transition matrix can be implemented in a dense, or a sparse, manner. If the
	transition matrix is dense, implementing it in a dense manner allows for
	the primary computation to use matrix multiplications which can be very
	fast. However, if the matrix is sparse, these matrix multiplications will
	be fairly slow and end up significantly slower than the sparse version of
	a matrix multiplication.

	This object is a wrapper for both implementations, which can be specified
	using the `kind` parameter. Choosing the right implementation will not
	effect the accuracy of the results but will change the speed at which they
	are calculated. 	

	Separately, there are two ways to instantiate the hidden Markov model. The
	first is by passing in a set of distributions, a dense transition matrix, 
	and optionally start/end probabilities. The second is to initialize the
	object without these and then to add edges using the `add_edge` method
	and to add nodes using the `add_nodes` method. Importantly, the way that
	you choose to initialize the hidden Markov model is independent of the
	implementation that you end up choosing. If you pass in a dense transition
	matrix, this will be converted to a sparse matrix with all the zeros
	dropped if you choose `kind='sparse'`.


	Parameters
	----------
	distributions: tuple or list
		A set of distribution objects. These objects do not need to be
		initialized, i.e., can be "Normal()". 

	edges: numpy.ndarray, torch.Tensor, or None. shape=(k,k), optional
		A dense transition matrix of probabilities for how each node or
		distribution passed in connects to each other one. This can contain
		many zeroes, and when paired with `kind='sparse'`, will drop those
		elements from the matrix. Default is None.

	starts: list, numpy.ndarray, torch.Tensor, or None. shape=(k,), optional
		The probability of starting at each node. If not provided, assumes
		these probabilities are uniform. Default is None.

	ends: list, numpy.ndarray, torch.Tensor, or None. shape=(k,), optional
		The probability of ending at each node. If not provided, assumes
		these probabilities are uniform. Default is None.

	kind: str, 'sparse' or 'dense', optional
		The underlying implementation of the transition matrix to use.
		Default is 'sparse'. 

	init: str, optional
		The initialization to use for the k-means initialization approach.
		Default is 'first-k'. Must be one of:

			'first-k': Use the first k examples from the data set
			'random': Use a random set of k examples from the data set
			'submodular-facility-location': Use a facility location submodular
				objective to initialize the k-means algorithm
			'submodular-feature-based': Use a feature-based submodular objective
				to initialize the k-means algorithm.

	max_iter: int, optional
		The number of iterations to do in the EM step, which for HMMs is
		sometimes called Baum-Welch. Default is 10.

	tol: float, optional
		The threshold at which to stop during fitting when the improvement
		goes under. Default is 0.1.

	inertia: float, [0, 1], optional
		Indicates the proportion of the update to apply to the parameters
		during training. When the inertia is 0.0, the update is applied in
		its entirety and the previous parameters are ignored. When the
		inertia is 1.0, the update is entirely ignored and the previous
		parameters are kept, equivalently to if the parameters were frozen.

	frozen: bool, optional
		Whether all the parameters associated with this distribution are frozen.
		If you want to freeze individual pameters, or individual values in those
		parameters, you must modify the `frozen` attribute of the tensor or
		parameter directly. Default is False.

	random_state: int or None, optional
		The random state to make randomness deterministic. If None, not
		deterministic. Default is None.

	verbose: bool, optional
		Whether to print the improvement and timings during training.
	"""

	def __init__(self, distributions=None, edges=None, starts=None, ends=None, 
		init='random', max_iter=1000, tol=0.1, sample_length=None, 
		return_sample_paths=False, inertia=0.0, frozen=False, 
		random_state=None, verbose=False):
		super().__init__(inertia=inertia, frozen=frozen)

		self.distributions = distributions or []

		n = len(distributions) if distributions is not None else None
		self.n_distributions = n
		self.n_edges = len(edges) if edges is not None else None

		self.edges = _check_parameter(_cast_as_tensor(edges), "edges",
			ndim=2, shape=(n, n), min_value=0., max_value=1.)
		self.starts = _check_parameter(_cast_as_tensor(starts), "starts",
			ndim=1, shape=(n,), min_value=0., max_value=1., value_sum=1.0)
		self.ends = _check_parameter(_cast_as_tensor(ends), "ends",
			ndim=1, shape=(n,), min_value=0., max_value=1.)

		if self.edges is None and distributions is not None:
			self.edges = torch.ones(n, n, dtype=torch.float32) / n
		elif self.edges is None and distributions is None:
			self.edges = []

		self.start = Silent()
		self.end = Silent()

		if self.starts is None and distributions is not None:
			self.starts = torch.ones(n, dtype=torch.float32) / n

		if self.ends is None and distributions is not None:
			self.ends = torch.ones(n, dtype=torch.float32) / n

		if not isinstance(random_state, numpy.random.RandomState):
			self.random_state = numpy.random.RandomState(random_state)
		else:
			self.random_state = random_state

		self.init = init
		self.max_iter = _check_parameter(max_iter, "max_iter", min_value=1, 
			ndim=0, dtypes=(int, torch.int32, torch.int64))
		self.tol = _check_parameter(tol, "tol", min_value=0., ndim=0)

		self.sample_length = sample_length
		self.return_sample_paths = return_sample_paths

		self.verbose = verbose

		self.d = self.distributions[0].d if distributions is not None else None
		self._initialized = all(n._initialized for n in self.distributions)

	def _initialize(self, X, sample_weight=None):
		"""Initialize the probability distribution.

		This method is meant to only be called internally. It initializes the
		parameters of the distribution and stores its dimensionality. For more
		complex methods, this function will do more.


		Parameters
		----------
		X: list, numpy.ndarray, torch.Tensor, shape=(-1, len, self.d)
			The data to use to initialize the model.

		sample_weight: list, tuple, numpy.ndarray, torch.Tensor, optional
			A set of weights for the examples. This can be either of shape
			(-1, len) or a vector of shape (-1,). Default is ones.

		priors: list, numpy.ndarray, torch.Tensor, shape=(-1, len, self.n)
			Prior probabilities of assigning each symbol to each node. If not
			provided, do not include in the calculations (conceptually
			equivalent to a uniform probability, but without scaling the
			probabilities).
		"""

		X = _check_parameter(_cast_as_tensor(X), "X", ndim=3)
		X = X.reshape(-1, X.shape[-1])

		if sample_weight is None:
			sample_weight = torch.ones(1).expand(X.shape[0], 1)
		else:
			sample_weight = _cast_as_tensor(sample_weight).reshape(-1, 1)
			sample_weight = _check_parameter(sample_weight, "sample_weight", 
				min_value=0., ndim=1, shape=(len(X),)).reshape(-1, 1)

		y_hat = KMeans(self.n_distributions, init=self.init, max_iter=1, 
			random_state=self.random_state).fit_predict(X, 
			sample_weight=sample_weight)

		for i in range(self.n_distributions):
			self.distributions[i].fit(X[y_hat == i], 
				sample_weight=sample_weight[y_hat == i])

		self._initialized = True
		self._reset_cache()
		self.d = X.shape[-1]
		super()._initialize(X.shape[-1])


	def _emission_matrix(self, X):
		"""Return the emission/responsibility matrix.

		This method returns the log probability of each example under each
		distribution contained in the model with the log prior probability
		of each component added.


		Parameters
		----------
		X: list, numpy.ndarray, torch.Tensor, shape=(-1, len, self.d)
			A set of examples to evaluate. 

	
		Returns
		-------
		e: torch.Tensor, shape=(-1, len, self.k)
			A set of log probabilities for each example under each distribution.
		"""

		X = _check_parameter(_cast_as_tensor(X), "X", ndim=3, 
			shape=(-1, -1, self.d))

		n, k, _ = X.shape
		X = X.reshape(n*k, self.d)

		e = torch.empty((k, self.n_distributions, n), dtype=self.dtype, 
			requires_grad=False, device=self.device)
		
		for i, node in enumerate(self.distributions):
			logp = node.log_probability(X)
			if isinstance(logp, torch.masked.MaskedTensor):
				logp = logp._masked_data

			e[:, i] = logp.reshape(n, k).T

		return e.permute(2, 0, 1)

	def log_probability(self, X, priors=None, check_inputs=True):
		"""Calculate the log probability of each example.

		This method calculates the log probability of each example given the
		parameters of the distribution. The examples must be given in a 3D
		format.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)
			A set of examples to evaluate.

		priors: list, numpy.ndarray, torch.Tensor, shape=(-1, length, self.d)
			Prior probabilities of assigning each symbol to each node. If not
			provided, do not include in the calculations (conceptually
			equivalent to a uniform probability, but without scaling the
			probabilities).

		check_inputs: bool, optional
			Whether to check the shape of the inputs and calculate emission
			matrices. Default is True.


		Returns
		-------
		logp: torch.Tensor, shape=(-1,)
			The log probability of each example.
		"""

		f = self.forward(X, priors=priors, check_inputs=check_inputs)
		return torch.logsumexp(f[:, -1] + self.ends, dim=1)

	def predict_log_proba(self, X, priors=None):
		"""Calculate the posterior probabilities for each example.

		This method calculates the log posterior probabilities for each example
		and then normalizes across each component of the model. These
		probabilities are calculated using the forward-backward algorithm.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, len, self.d)
			A set of examples to summarize.

		priors: list, numpy.ndarray, torch.Tensor, shape=(-1, len, self.n_distributions)
			Prior probabilities of assigning each symbol to each node. If not
			provided, do not include in the calculations (conceptually
			equivalent to a uniform probability, but without scaling the
			probabilities).


		Returns
		-------
		r: torch.Tensor, shape=(-1, len, self.n_distributions)
			The log posterior probabilities for each example under each 
			component as calculated by the forward-backward algorithm.
		"""

		_, r, _, _, _ = self.forward_backward(X, priors=priors)
		return r

	def predict_proba(self, X, priors=None):
		"""Calculate the posterior probabilities for each example.

		This method calculates the posterior probabilities for each example
		and then normalizes across each component of the model. These
		probabilities are calculated using the forward-backward algorithm.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, len, self.d)
			A set of examples to summarize.

		priors: list, numpy.ndarray, torch.Tensor, shape=(-1, len, self.n_distributions)
			Prior probabilities of assigning each symbol to each node. If not
			provided, do not include in the calculations (conceptually
			equivalent to a uniform probability, but without scaling the
			probabilities).


		Returns
		-------
		y: torch.Tensor, shape=(-1, len, self.n_distributions)
			The posterior probabilities for each example under each component
			as calculated by the forward-backward algorithm.
		"""

		return torch.exp(self.predict_log_proba(X, priors=priors))

	def predict(self, X, priors=None):
		"""Predicts the component for each observation.

		This method calculates the predicted component for each observation
		given the posterior probabilities as calculated by the forward-backward
		algorithm. Essentially, it is just the argmax over components.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, len, self.d)
			A set of examples to summarize.

		priors: list, numpy.ndarray, torch.Tensor, shape=(-1, length, self.d)
			Prior probabilities of assigning each symbol to each node. If not
			provided, do not include in the calculations (conceptually
			equivalent to a uniform probability, but without scaling the
			probabilities).


		Returns
		-------
		y: torch.Tensor, shape=(-1, len, self.k)
			The posterior probabilities for each example under each component
			as calculated by the forward-backward algorithm.
		"""

		return torch.argmax(self.predict_log_proba(X, priors=priors), dim=-1)

	def fit(self, X, sample_weight=None, priors=None):
		"""Fit the model to optionally weighted examples.

		This method implements the core of the learning process. For a hidden
		Markov model, this involves performing EM until the distributions that
		are being fit converge according to the threshold set by `tol`, or
		until the maximum number of iterations has been hit. Sometimes, this
		is called the Baum-Welch algorithm.

		This method is largely a wrapper around the `summarize` and
		`from_summaries` methods. It's primary contribution is serving as a
		loop around these functions and to monitor convergence.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, len, self.d)
			A set of examples to evaluate. 

		y: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, len), optional 
			A set of labels with the same number of examples and length as the
			observations that indicate which node in the model that each
			observation should be assigned to. Passing this in means that the
			model uses labeled training instead of Baum-Welch. Default is None.

		sample_weight: list, tuple, numpy.ndarray, torch.Tensor, optional
			A set of weights for the examples. This can be either of shape
			(-1, self.d) or a vector of shape (-1,). Default is ones.


		Returns
		-------
		self
		"""

		if not self._initialized:
			self._initialize(X, sample_weight=sample_weight)

		logp, last_logp = None, None
		for i in range(self.max_iter):
			start_time = time.time()
			logp = self.summarize(X, sample_weight=sample_weight, 
				priors=priors).sum()

			if i > 0:
				improvement = logp - last_logp
				duration = time.time() - start_time

				if self.verbose:
					print("[{}] Improvement: {}, Time: {:4.4}s".format(i, 
						improvement, duration))

				if improvement < self.tol:
					self._reset_cache()
					return self

			last_logp = logp
			self.from_summaries()

		if self.verbose:
			logp = self.summarize(X, sample_weight=sample_weight, 
				priors=priors).sum()

			improvement = logp - last_logp
			duration = time.time() - start_time

			print("[{}] Improvement: {}, Time: {:4.4}s".format(i+1, 
				improvement, duration))

		self._reset_cache()
		return self

	def summarize(self, X, sample_weight=None, emissions=None, 
		priors=None):
		"""Extract the sufficient statistics from a batch of data.

		This method calculates the sufficient statistics from optionally
		weighted data and adds them to the stored cache. The examples must be
		given in a 2D format. Sample weights can either be provided as one
		value per example or as a 2D matrix of weights for each feature in
		each example.


		Parameters
		----------
		X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, len, self.d)
			A set of examples to summarize.

		sample_weight: list, tuple, numpy.ndarray, torch.Tensor, optional
			A set of weights for the examples. This can be either of shape
			(-1, length, self.d) or a vector of shape (-1,). Default is ones.

		emissions: list, numpy.ndarray, torch.Tensor, shape=(-1, -1, n_distributions)
			Precalculated emission log probabilities. These are the
			probabilities of each observation under each probability 
			distribution. When running some algorithms it is more efficient
			to precalculate these and pass them into each call.

		priors: list, numpy.ndarray, torch.Tensor, shape=(-1, len, self.n_distributions)
			Prior probabilities of assigning each symbol to each node. If not
			provided, do not include in the calculations (conceptually
			equivalent to a uniform probability, but without scaling the
			probabilities).


		Returns
		-------
		logp: torch.Tensor, shape=(-1,)
			The log probability of each example.
		"""

		X = _check_parameter(_cast_as_tensor(X), "X", ndim=3, 
			shape=(-1, -1, self.d))
		emissions, priors = _check_inputs(self, X, emissions, priors)
		
		if sample_weight is None:
			sample_weight = torch.ones(1, device=self.device).expand(
				emissions.shape[0], 1)
		else:
			sample_weight = _check_parameter(_cast_as_tensor(sample_weight),
				"sample_weight", min_value=0., ndim=1, 
				shape=(emissions.shape[0],)).reshape(-1, 1)

		if not self._initialized and y is None:
			self._initialize(X, sample_weight=sample_weight)

		return X, emissions, priors, sample_weight

	def from_summaries(self):
		"""Update the model parameters given the extracted statistics.

		This method uses calculated statistics from calls to the `summarize`
		method to update the distribution parameters. Hyperparameters for the
		update are passed in at initialization time.

		Note: Internally, a call to `fit` is just a successive call to the
		`summarize` method followed by the `from_summaries` method.
		"""

		self.from_summaries()
		self._reset_cache()


