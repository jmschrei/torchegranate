# test_exponential.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy
import torch
import pytest

from torchegranate.distributions import Exponential

from nose.tools import assert_raises
from numpy.testing import assert_array_almost_equal

@pytest.fixture
def X():
	return [[1, 2, 0],
	     [0, 0, 1],
	     [1, 1, 2],
	     [2, 2, 2],
	     [3, 1, 0],
	     [5, 1, 4],
	     [2, 1, 0]]


@pytest.fixture
def X2():
	return [[1.2, 0.5, 1.1, 1.9],
	     [6.2, 1.1, 2.4, 1.1]] 


@pytest.fixture
def w():
	return [[1], [2], [0], [0], [5], [1], [2]]


@pytest.fixture
def w2():
	return [[1.1], [3.5]]


###


def _test_initialization(d, x, inertia, frozen, dtype):
	assert d.inertia == inertia
	assert d.frozen == frozen

	if x is not None:
		assert d.rates.shape == (len(x),)
		assert d.rates.dtype == dtype
		assert_array_almost_equal(d.rates, x)
	else:
		assert d.rates == x


def test_initialization():
	d = Exponential()
	_test_initialization(d, None, 0.0, False, None)
	assert_raises(AttributeError, getattr, d, "_w_sum")
	assert_raises(AttributeError, getattr, d, "_xw_sum")
	assert_raises(AttributeError, getattr, d, "_log_rates")


def test_initialization_int():
	funcs = (lambda x: x, tuple, numpy.array, torch.tensor, 
		lambda x: torch.nn.Parameter(torch.tensor(x), requires_grad=False))

	x = [1, 2, 3, 8, 1]
	for func in funcs:
		y = func(x)
		_test_initialization(Exponential(y, inertia=0.0, frozen=False), 
			y, 0.0, False, torch.int64)
		_test_initialization(Exponential(y, inertia=0.3, frozen=False), 
			y, 0.3, False, torch.int64)
		_test_initialization(Exponential(y, inertia=1.0, frozen=True), 
			y, 1.0, True, torch.int64)
		_test_initialization(Exponential(y, inertia=1.0, frozen=False), 
			y, 1.0, False, torch.int64)

	x = numpy.array(x, dtype=numpy.int32)
	for func in funcs[2:]:
		y = func(x)
		_test_initialization(Exponential(y, inertia=0.0, frozen=False), 
			y, 0.0, False, torch.int32)
		_test_initialization(Exponential(y, inertia=0.3, frozen=False), 
			y, 0.3, False, torch.int32)
		_test_initialization(Exponential(y, inertia=1.0, frozen=True), 
			y, 1.0, True, torch.int32)
		_test_initialization(Exponential(y, inertia=1.0, frozen=False), 
			y, 1.0, False, torch.int32)


def test_initialization_float():
	funcs = (lambda x: x, tuple, 
		lambda x: numpy.array(x, dtype=numpy.float32), 
		lambda x: torch.tensor(x, dtype=torch.float32), 
		lambda x: torch.nn.Parameter(torch.tensor(x), requires_grad=False))

	x = [1.0, 2.2, 3.9, 8.1, 1.0]
	for func in funcs:
		y = func(x)
		_test_initialization(Exponential(y, inertia=0.0, frozen=False), 
			y, 0.0, False, torch.float32)
		_test_initialization(Exponential(y, inertia=0.3, frozen=False), 
			y, 0.3, False, torch.float32)
		_test_initialization(Exponential(y, inertia=1.0, frozen=True), 
			y, 1.0, True, torch.float32)
		_test_initialization(Exponential(y, inertia=1.0, frozen=False), 
			y, 1.0, False, torch.float32)

	x = numpy.array(x, dtype=numpy.float64)
	_test_initialization(Exponential(x, inertia=0.0, frozen=False), 
		x, 0.0, False, torch.float64)
	_test_initialization(Exponential(x, inertia=0.3, frozen=False), 
		x, 0.3, False, torch.float64)
	_test_initialization(Exponential(x, inertia=1.0, frozen=True), 
		x, 1.0, True, torch.float64)
	_test_initialization(Exponential(x, inertia=1.0, frozen=False), 
		x, 1.0, False, torch.float64)


def test_initialization_raises():
	assert_raises(ValueError, Exponential, 1)
	assert_raises(ValueError, Exponential, 1.2)
	assert_raises(ValueError, Exponential, [0], inertia=-0.4)
	assert_raises(ValueError, Exponential, [0.2], inertia=-0.4)
	assert_raises(ValueError, Exponential, [0], inertia=1.2)
	assert_raises(ValueError, Exponential, [0.9], inertia=1.2)
	assert_raises(ValueError, Exponential, frozen=3)
	assert_raises(ValueError, Exponential, frozen="true")


def test_reset_cache(X):
	d = Exponential()
	d.summarize(X)
	assert_array_almost_equal(d._w_sum, [7.0, 7.0, 7.0])
	assert_array_almost_equal(d._xw_sum, [14.0, 8.0, 9.0])

	d._reset_cache()
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0, 0.0])	

	d = Exponential()
	assert_raises(AttributeError, getattr, d, "_w_sum")
	assert_raises(AttributeError, getattr, d, "_xw_sum")
	assert_raises(AttributeError, getattr, d, "_log_rates")

	d._reset_cache()
	assert_raises(AttributeError, getattr, d, "_w_sum")
	assert_raises(AttributeError, getattr, d, "_xw_sum")
	assert_raises(AttributeError, getattr, d, "_log_rates")


def test_initialize(X):
	d = Exponential()
	assert d.d is None
	assert d.rates is None
	assert d._initialized == False
	assert_raises(AttributeError, getattr, d, "_w_sum")
	assert_raises(AttributeError, getattr, d, "_xw_sum")
	assert_raises(AttributeError, getattr, d, "_log_rates")

	d._initialize(3)
	assert d._initialized == True
	assert d.rates.shape[0] == 3
	assert d.d == 3
	assert_array_almost_equal(d.rates, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0, 0.0])	

	d._initialize(2)
	assert d._initialized == True
	assert d.rates.shape[0] == 2
	assert d.d == 2
	assert_array_almost_equal(d.rates, [0.0, 0.0])
	assert_array_almost_equal(d._w_sum, [0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0])	

	d = Exponential([1.2, 9.3])
	assert d._initialized == True
	assert d.d == 2

	d._initialize(3)
	assert d._initialized == True
	assert d.rates.shape[0] == 3
	assert d.d == 3
	assert_array_almost_equal(d.rates, [0.0, 0.0, 0.0])

	d = Exponential()
	d.summarize(X)
	d._initialize(4)
	assert d._initialized == True
	assert d.rates.shape[0] == 4
	assert d.d == 4
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0, 0.0, 0.0])	


###


def _test_predictions(x, y, y_hat, dtype):
	assert isinstance(y_hat, torch.Tensor)
	assert y_hat.dtype == dtype
	assert y_hat.shape == (len(x),)
	assert_array_almost_equal(y, y_hat)


def test_probability():
	p = [1.7]
	d1 = Exponential(p)
	d2 = Exponential(numpy.array(p, dtype=numpy.float64))
	x = [[1.0], [2.0], [8.0], [3.7], [1.9]]
	y = [3.105620e-01, 5.673455e-02, 2.108841e-06, 3.153090e-03, 6.724774e-02]
	_test_predictions(x, y, d1.probability(x), torch.float32)
	_test_predictions(x, y, d2.probability(x), torch.float64)

	p = [1.7, 2.3, 0.1, 0.8, 4.1]
	d1 = Exponential(p)
	d2 = Exponential(numpy.array(p, dtype=numpy.float64))
	x = [[1.0, 2.0, 8.0, 3.7, 1.9]]
	y = torch.prod(torch.tensor([3.105620e-01, 5.673455e-02, 2.108841e-06, 
		3.153090e-03, 6.724774e-02])).reshape(1,)
	_test_predictions(x, y, d1.probability(x), torch.float32)
	_test_predictions(x, y, d2.probability(x), torch.float64)

	p = [1, 2, 4]
	d1 = Exponential(p)
	d2 = Exponential(numpy.array(p, dtype=numpy.float64))
	x = [[1, 2, 1],
	     [2, 2, 1],
	     [0, 1, 0],
	     [0, 0, 2]]
	y = [9.872784e-04, 3.631994e-04, 1.082682e+00, 2.683701e-03]
	_test_predictions(x, y, d1.probability(x), torch.float32)
	_test_predictions(x, y, d2.probability(x), torch.float64)

	p = [1.0, 2.0, 4.0]
	d1 = Exponential(p)
	d2 = Exponential(numpy.array(p, dtype=numpy.float64))
	_test_predictions(x, y, d1.probability(x), torch.float32)
	_test_predictions(x, y, d2.probability(x), torch.float64)


def test_probability_dtypes():
	X = numpy.random.uniform(0, 5, size=(10, 3)).astype(numpy.float32)
	X_int = X.astype('int32')
	p = numpy.array([0.1, 5.3, 2.5], dtype=numpy.float32)
	
	y = Exponential(p).probability(X)
	assert y.dtype == torch.float32

	y = Exponential(p).probability(X_int)
	assert y.dtype == torch.float32

	X = numpy.random.uniform(0, 5, size=(10, 3)).astype(numpy.float64)
	X_int = X.astype('int32')
	p = numpy.array([0.1, 5.3, 2.5], dtype=numpy.float64)

	y = Exponential(p).probability(X)
	assert y.dtype == torch.float64

	y = Exponential(p).probability(X_int)
	assert y.dtype == torch.float64


def test_probability_raises():
	d = Exponential([1.2, 1.8, 2.1])
	assert_raises(ValueError, d.probability, [[1.1, 1.2, 1.9, 1.2]])
	assert_raises(ValueError, d.probability, [[1.1]])
	assert_raises(ValueError, d.probability, [1.1, 1.2, 1.9, 1.2])
	assert_raises(ValueError, d.probability, [1.1])
	assert_raises(ValueError, d.probability, [1.1, 1.2, 1.9])
	assert_raises(ValueError, d.probability, [[[1.1]]])

	d = Exponential([1.2])
	assert_raises(ValueError, d.probability, [[1.1, 1.2, 1.9, 1.2]])
	assert_raises(ValueError, d.probability, [[]])
	assert_raises(ValueError, d.probability, [1.1, 1.2, 1.9, 1.2])
	assert_raises(ValueError, d.probability, [1.1])
	assert_raises(ValueError, d.probability, [[[1.1]]])


def test_log_probability():
	p = [1.7]
	d1 = Exponential(p)
	d2 = Exponential(numpy.array(p, dtype=numpy.float64))
	x = [[1.0], [2.0], [8.0], [3.7], [1.9]]
	y = torch.log(torch.tensor([3.105620e-01, 5.673455e-02, 2.108841e-06, 
		3.153090e-03, 6.724774e-02]))
	_test_predictions(x, y, d1.log_probability(x), torch.float32)
	_test_predictions(x, y, d2.log_probability(x), torch.float64)

	p_torch = torch.tensor(numpy.array(p))
	d3 = torch.distributions.Exponential(p_torch)
	x_torch = torch.tensor(numpy.array(x))
	_test_predictions(x, d3.log_prob(x_torch).sum(axis=1), 
		d2.log_probability(x), torch.float64)

	p = [1.7, 2.3, 0.1, 0.8, 4.1]
	d1 = Exponential(p)
	d2 = Exponential(numpy.array(p, dtype=numpy.float64))
	d3 = torch.distributions.Exponential(p_torch)
	x = [[1.0, 2.0, 8.0, 3.7, 1.9]]
	y = [-17.601204]
	_test_predictions(x, y, d1.log_probability(x), torch.float32)
	_test_predictions(x, y, d2.log_probability(x), torch.float64)

	p_torch = torch.tensor(numpy.array(p))
	d3 = torch.distributions.Exponential(p_torch)
	x_torch = torch.tensor(numpy.array(x))
	_test_predictions(x, d3.log_prob(x_torch).sum(axis=1), 
		d2.log_probability(x), torch.float64)

	p = [1, 2, 4]
	d1 = Exponential(p)
	d2 = Exponential(numpy.array(p, dtype=numpy.float64))
	d3 = torch.distributions.Exponential(p_torch)
	x = [[1, 2, 1],
	     [2, 2, 1],
	     [0, 1, 0],
	     [0, 0, 2]]
	y = torch.log(torch.tensor([9.872784e-04, 3.631994e-04, 1.082682e+00, 
		2.683701e-03]))
	_test_predictions(x, y, d1.log_probability(x), torch.float32)
	_test_predictions(x, y, d2.log_probability(x), torch.float64)

	p = [1.0, 2.0, 4.0]
	d1 = Exponential(p)
	d2 = Exponential(numpy.array(p, dtype=numpy.float64))
	d3 = torch.distributions.Exponential(p_torch)
	_test_predictions(x, y, d1.log_probability(x), torch.float32)
	_test_predictions(x, y, d2.log_probability(x), torch.float64)

def test_log_probability_dtypes():
	X = numpy.random.uniform(0, 5, size=(10, 3)).astype(numpy.float32)
	X_int = X.astype('int32')
	p = numpy.array([0.1, 5.3, 2.5], dtype=numpy.float32)
	
	y = Exponential(p).log_probability(X)
	assert y.dtype == torch.float32

	y = Exponential(p).log_probability(X_int)
	assert y.dtype == torch.float32

	X = numpy.random.uniform(0, 5, size=(10, 3)).astype(numpy.float64)
	X_int = X.astype('int32')
	p = numpy.array([0.1, 5.3, 2.5], dtype=numpy.float64)

	y = Exponential(p).log_probability(X)
	assert y.dtype == torch.float64

	y = Exponential(p).log_probability(X_int)
	assert y.dtype == torch.float64


def test_log_probability_raises():
	d = Exponential([1.2, 1.8, 2.1])
	assert_raises(ValueError, d.log_probability, [[1.1, 1.2, 1.9, 1.2]])
	assert_raises(ValueError, d.log_probability, [[1.1]])
	assert_raises(ValueError, d.log_probability, [1.1, 1.2, 1.9, 1.2])
	assert_raises(ValueError, d.log_probability, [1.1])
	assert_raises(ValueError, d.log_probability, [1.1, 1.2, 1.9])
	assert_raises(ValueError, d.log_probability, [[[1.1]]])

	d = Exponential([1.2])
	assert_raises(ValueError, d.log_probability, [[1.1, 1.2, 1.9, 1.2]])
	assert_raises(ValueError, d.log_probability, [[]])
	assert_raises(ValueError, d.log_probability, [1.1, 1.2, 1.9, 1.2])
	assert_raises(ValueError, d.log_probability, [1.1])
	assert_raises(ValueError, d.log_probability, [[[1.1]]])


###


def test_summarize(X, X2):
	d = Exponential([1.3, 2.3, 6.1])
	d.summarize(X[:4])
	assert_array_almost_equal(d._w_sum, [4.0, 4.0, 4.0])
	assert_array_almost_equal(d._xw_sum, [4.0, 5.0, 5.0])

	d.summarize(X[4:])
	assert_array_almost_equal(d._w_sum, [7.0, 7.0, 7.0])
	assert_array_almost_equal(d._xw_sum, [14.0, 8.0, 9.0])

	d = Exponential([1.3, 2.3, 6.1])
	d.summarize(X)
	assert_array_almost_equal(d._w_sum, [7.0, 7.0, 7.0])
	assert_array_almost_equal(d._xw_sum, [14.0, 8.0, 9.0])


	d = Exponential()
	d.summarize(X[:4])
	assert_array_almost_equal(d._w_sum, [4.0, 4.0, 4.0])
	assert_array_almost_equal(d._xw_sum, [4.0, 5.0, 5.0])

	d.summarize(X[4:])
	assert_array_almost_equal(d._w_sum, [7.0, 7.0, 7.0])
	assert_array_almost_equal(d._xw_sum, [14.0, 8.0, 9.0])

	d = Exponential()
	d.summarize(X)
	assert_array_almost_equal(d._w_sum, [7.0, 7.0, 7.0])
	assert_array_almost_equal(d._xw_sum, [14.0, 8.0, 9.0])


def test_summarize_weighted(X, X2, w, w2):
	d = Exponential([1.3, 2.3, 6.1])
	d.summarize(X[:4], sample_weights=w[:4])
	assert_array_almost_equal(d._w_sum, [3., 3., 3.])
	assert_array_almost_equal(d._xw_sum, [1., 2., 2.])

	d.summarize(X[4:], sample_weights=w[4:])
	assert_array_almost_equal(d._w_sum, [11.0, 11.0, 11.0])
	assert_array_almost_equal(d._xw_sum, [25.0, 10.0, 6.0])

	d = Exponential([1.3, 2.3, 6.1])
	d.summarize(X, sample_weights=w)
	assert_array_almost_equal(d._w_sum, [11.0, 11.0, 11.0])
	assert_array_almost_equal(d._xw_sum, [25.0, 10.0, 6.0])


	d = Exponential()
	d.summarize(X2, sample_weights=w2)
	assert_array_almost_equal(d._w_sum, [4.6, 4.6, 4.6, 4.6])
	assert_array_almost_equal(d._xw_sum, [23.02, 4.4, 9.61, 5.94])

	d = Exponential([0, 0, 0, 0])
	d.summarize(X2, sample_weights=w2)
	assert_array_almost_equal(d._w_sum, [4.6, 4.6, 4.6, 4.6])
	assert_array_almost_equal(d._xw_sum, [23.02, 4.4, 9.61, 5.94])


def test_summarize_weighted_flat(X, X2, w, w2):
	w = numpy.array(w)[:,0] 

	d = Exponential([1.3, 2.3, 6.1])
	d.summarize(X[:4], sample_weights=w[:4])
	assert_array_almost_equal(d._w_sum, [3., 3., 3.])
	assert_array_almost_equal(d._xw_sum, [1., 2., 2.])

	d.summarize(X[4:], sample_weights=w[4:])
	assert_array_almost_equal(d._w_sum, [11.0, 11.0, 11.0])
	assert_array_almost_equal(d._xw_sum, [25.0, 10.0, 6.0])

	d = Exponential([1.3, 2.3, 6.1])
	d.summarize(X, sample_weights=w)
	assert_array_almost_equal(d._w_sum, [11.0, 11.0, 11.0])
	assert_array_almost_equal(d._xw_sum, [25.0, 10.0, 6.0])


	d = Exponential()
	d.summarize(X2, sample_weights=w2)
	assert_array_almost_equal(d._w_sum, [4.6, 4.6, 4.6, 4.6])
	assert_array_almost_equal(d._xw_sum, [23.02, 4.4, 9.61, 5.94])

	d = Exponential([0, 0, 0, 0])
	d.summarize(X2, sample_weights=w2)
	assert_array_almost_equal(d._w_sum, [4.6, 4.6, 4.6, 4.6])
	assert_array_almost_equal(d._xw_sum, [23.02, 4.4, 9.61, 5.94])


def test_summarize_weighted_2d(X):
	d = Exponential()
	d.summarize(X[:4], sample_weights=X[:4])
	assert_array_almost_equal(d._w_sum, [4., 5., 5.])
	assert_array_almost_equal(d._xw_sum, [6., 9., 9.])

	d.summarize(X[4:], sample_weights=X[4:])
	assert_array_almost_equal(d._w_sum, [14., 8., 9.])
	assert_array_almost_equal(d._xw_sum, [44., 12., 25.])

	d = Exponential()
	d.summarize(X, sample_weights=X)
	assert_array_almost_equal(d._w_sum, [14., 8., 9.])
	assert_array_almost_equal(d._xw_sum, [44., 12., 25.])


def test_summarize_dtypes(X, w):
	X = numpy.array(X)
	X = X.astype(numpy.float32)

	p = numpy.array([3.0, 1.1, 2.8], dtype=numpy.float32)
	d = Exponential(p)
	assert d._xw_sum.dtype == torch.float32
	d.summarize(X)
	assert d._xw_sum.dtype == torch.float32

	X = X.astype(numpy.float64)
	p = numpy.array([3.0, 1.1, 2.8], dtype=numpy.float32)
	d = Exponential(p)
	assert d._xw_sum.dtype == torch.float32
	d.summarize(X)
	assert d._xw_sum.dtype == torch.float32

	X = X.astype(numpy.int32)
	p = numpy.array([3.0, 1.1, 2.8], dtype=numpy.float32)
	d = Exponential(p)
	assert d._xw_sum.dtype == torch.float32
	d.summarize(X)
	assert d._xw_sum.dtype == torch.float32

	X = X.astype(numpy.int64)
	p = numpy.array([3.0, 1.1, 2.8], dtype=numpy.float32)
	d = Exponential(p)
	assert d._xw_sum.dtype == torch.float32
	d.summarize(X)
	assert d._xw_sum.dtype == torch.float32


def test_summarize_raises(X, w):
	d = Exponential([1.2, 1.8, 2.1])
	assert_raises(ValueError, d.summarize, [[1.1, 1.2, 1.9, 1.2]])
	assert_raises(ValueError, d.summarize, [[1.1]])
	assert_raises(ValueError, d.summarize, [1.1, 1.2, 1.9, 1.2])
	assert_raises(ValueError, d.summarize, [1.1])
	assert_raises(ValueError, d.summarize, [1.1, 1.2, 1.9])
	assert_raises(ValueError, d.summarize, [[[1.1]]])

	d = Exponential([1.2])
	assert_raises(ValueError, d.summarize, [[1.1, 1.2, 1.9, 1.2]])
	assert_raises(ValueError, d.summarize, [[]])
	assert_raises(ValueError, d.summarize, [1.1, 1.2, 1.9, 1.2])
	assert_raises(ValueError, d.summarize, [1.1])
	assert_raises(ValueError, d.summarize, [[[1.1]]])

	d = Exponential([1.2, 1.8, 2.1])
	assert_raises(ValueError, d.summarize, [X])
	assert_raises(ValueError, d.summarize, [X], w)
	assert_raises(ValueError, d.summarize, X, [w])
	assert_raises(ValueError, d.summarize, X, w[:3])
	assert_raises(ValueError, d.summarize, X[:3], w)


def test_from_summaries(X):
	d = Exponential([1.3, 2.3, 6.1])
	d.summarize(X[:4])
	d.from_summaries()
	assert_array_almost_equal(d.rates, [1.0, 0.8, 0.8])
	assert_array_almost_equal(d._log_rates, [0.0, -0.223144, -0.223144])
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0, 0.0])

	d.summarize(X[4:])
	d.from_summaries()
	assert_array_almost_equal(d.rates, [0.3, 1.0, 0.75])
	assert_array_almost_equal(d._log_rates, [-1.203973,  0.0, -0.287682])
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0, 0.0])

	d = Exponential([1.3, 2.3, 6.1])
	d.summarize(X[:4])
	d.summarize(X[4:])
	d.from_summaries()
	assert_array_almost_equal(d.rates, [0.5, 0.875, 0.777778])
	assert_array_almost_equal(d._log_rates, [-0.693147, -0.133531, -0.251314])
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0, 0.0])

	d = Exponential([1.3, 2.3, 6.1])
	d.summarize(X)
	d.from_summaries()
	assert_array_almost_equal(d.rates, [0.5, 0.875, 0.777778])
	assert_array_almost_equal(d._log_rates, [-0.693147, -0.133531, -0.251314])
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0, 0.0])

	d = Exponential()
	d.summarize(X[:4])
	d.from_summaries()
	assert_array_almost_equal(d.rates, [1.0, 0.8, 0.8])
	assert_array_almost_equal(d._log_rates, [0.0, -0.223144, -0.223144])
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0, 0.0])

	d.summarize(X[4:])
	d.from_summaries()
	assert_array_almost_equal(d.rates, [0.3, 1.0, 0.75])
	assert_array_almost_equal(d._log_rates, [-1.203973,  0.0, -0.287682])
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0, 0.0])

	d = Exponential()
	d.summarize(X[:4])
	d.summarize(X[4:])
	d.from_summaries()
	assert_array_almost_equal(d.rates, [0.5, 0.875, 0.777778])
	assert_array_almost_equal(d._log_rates, [-0.693147, -0.133531, -0.251314])
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0, 0.0])

	d = Exponential()
	d.summarize(X)
	d.from_summaries()
	assert_array_almost_equal(d.rates, [0.5, 0.875, 0.777778])
	assert_array_almost_equal(d._log_rates, [-0.693147, -0.133531, -0.251314])
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0, 0.0])


def test_from_summaries_weighted(X, w):
	d = Exponential([1.3, 2.3, 6.1])
	d.summarize(X[:4], sample_weights=w[:4])
	d.from_summaries()
	assert_array_almost_equal(d.rates, [3.0, 1.5, 1.5])
	assert_array_almost_equal(d._log_rates, [1.098612, 0.405465, 0.405465])
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0, 0.0])

	d.summarize(X[4:], sample_weights=w[4:])
	d.from_summaries()
	assert_array_almost_equal(d.rates, [0.333333, 1. , 2.])
	assert_array_almost_equal(d._log_rates, [-1.098612, 0., 0.693147])
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0, 0.0])

	d = Exponential([1.3, 2.3, 6.1])
	d.summarize(X, sample_weights=w)
	d.from_summaries()
	assert_array_almost_equal(d.rates, [0.44, 1.1, 1.833333])
	assert_array_almost_equal(d._log_rates, [-0.820981,  0.09531, 0.606136])
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0, 0.0])


	d = Exponential()
	d.summarize(X[:4], sample_weights=w[:4])
	d.from_summaries()
	assert_array_almost_equal(d.rates, [3.0, 1.5, 1.5])
	assert_array_almost_equal(d._log_rates, [1.098612, 0.405465, 0.405465])
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0, 0.0])

	d.summarize(X[4:], sample_weights=w[4:])
	d.from_summaries()
	assert_array_almost_equal(d.rates, [0.333333, 1. , 2.])
	assert_array_almost_equal(d._log_rates, [-1.098612, 0., 0.693147])
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0, 0.0])

	d = Exponential()
	d.summarize(X, sample_weights=w)
	d.from_summaries()
	assert_array_almost_equal(d.rates, [0.44, 1.1, 1.833333])
	assert_array_almost_equal(d._log_rates, [-0.820981,  0.09531, 0.606136])
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0, 0.0])

	X = [[1.2, 0.5, 1.1, 1.9],
	     [6.2, 1.1, 2.4, 1.1]] 

	w = [[1.1], [3.5]]

	d = Exponential()
	d.summarize(X, sample_weights=w)
	d.from_summaries()
	assert_array_almost_equal(d.rates, [0.199826, 1.045455, 0.478668, 0.774411])
	assert_array_almost_equal(d._log_rates, [-1.610307,  0.044452, -0.736748, 
		-0.255653])
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0, 0.0, 0.0])


def test_from_summaries_null():
	d = Exponential([1, 2])
	d.from_summaries()
	assert d.rates[0] != 1 and d.rates[1] != 2 
	assert_array_almost_equal(d._w_sum, [0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0])

	d = Exponential([1, 2], inertia=0.5)
	d.from_summaries()
	assert d.rates[0] != 1 and d.rates[1] != 2 
	assert_array_almost_equal(d._w_sum, [0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0])

	d = Exponential([1, 2], inertia=0.5, frozen=True)
	d.from_summaries()
	assert_array_almost_equal(d.rates, [1, 2])
	assert_array_almost_equal(d._w_sum, [0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0])


def test_from_summaries_inertia(X, w):
	d = Exponential([1.3, 2.3, 6.1], inertia=0.3)
	d.summarize(X[:4])
	d.from_summaries()
	assert_array_almost_equal(d.rates, [1.09, 1.25, 2.39])
	assert_array_almost_equal(d._log_rates, [0.086178, 0.223144, 0.871293])
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0, 0.0])

	d.summarize(X[4:])
	d.from_summaries()
	assert_array_almost_equal(d.rates, [0.537, 1.075, 1.242])
	assert_array_almost_equal(d._log_rates, [-0.621757,  0.072321,  0.216723])
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0, 0.0])

	d = Exponential([1.3, 2.3, 6.1], inertia=0.3)
	d.summarize(X)
	d.from_summaries()
	assert_array_almost_equal(d.rates, [0.74, 1.3025, 2.374444])
	assert_array_almost_equal(d._log_rates, [-0.301105, 0.264286, 0.864763])
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0, 0.0])


def test_from_summaries_weighted_inertia(X, w):
	d = Exponential([1.3, 2.3, 6.1], inertia=0.3)
	d.summarize(X, sample_weights=w)
	d.from_summaries()
	assert_array_almost_equal(d.rates, [0.698, 1.46, 3.113333])
	assert_array_almost_equal(d._log_rates, [-0.359536, 0.378436, 1.135694])
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0, 0.0])

	r = [1.3, 2.3, 6.1]
	d = Exponential(r, inertia=1.0)
	d.summarize(X[:4])
	d.from_summaries()
	assert_array_almost_equal(d.rates, r)
	assert_array_almost_equal(d._log_rates, numpy.log(r))
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0, 0.0])

	d.summarize(X[4:])
	d.from_summaries()
	assert_array_almost_equal(d.rates, r)
	assert_array_almost_equal(d._log_rates, numpy.log(r))
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0, 0.0])

	d = Exponential([1.3, 2.3, 6.1], inertia=1.0)
	d.summarize(X)
	d.from_summaries()
	assert_array_almost_equal(d.rates, r)
	assert_array_almost_equal(d._log_rates, numpy.log(r))
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0, 0.0])


def test_from_summaries_frozen(X, w):
	p = [1.3, 2.3, 6.1]
	d = Exponential(p, frozen=True)
	d.summarize(X[:4])
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0, 0.0])

	d.from_summaries()
	assert_array_almost_equal(d.rates, p)
	assert_array_almost_equal(d._log_rates, numpy.log(p))
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0, 0.0])

	d.summarize(X[4:])
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0, 0.0])

	d.from_summaries()
	assert_array_almost_equal(d.rates, p)
	assert_array_almost_equal(d._log_rates, numpy.log(p))
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0, 0.0])

	d = Exponential([1.3, 2.3, 6.1], frozen=True)
	d.summarize(X)
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0, 0.0])

	d.from_summaries()
	assert_array_almost_equal(d.rates, p)
	assert_array_almost_equal(d._log_rates, numpy.log(p))
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0, 0.0])

	d = Exponential([1.3, 2.3, 6.1], frozen=True)
	d.summarize(X, sample_weights=w)
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0, 0.0])

	d.from_summaries()
	assert_array_almost_equal(d.rates, p)
	assert_array_almost_equal(d._log_rates, numpy.log(p))
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0, 0.0])


def test_from_summaries_dtypes(X):
	X = numpy.array(X)
	X = X.astype(numpy.float32)

	p = numpy.array([1.2, 4.1, 0.3], dtype=numpy.float32)
	d = Exponential(p)
	d.summarize(X)
	d.from_summaries()
	assert d.rates.dtype == torch.float32
	assert d._log_rates.dtype == torch.float32

	p = numpy.array([1.2, 4.1, 0.3], dtype=numpy.float64)
	d = Exponential(p)
	d.summarize(X)
	d.from_summaries()
	assert d.rates.dtype == torch.float64
	assert d._log_rates.dtype == torch.float64

	p = numpy.array([1, 4, 0], dtype=numpy.int32)
	d = Exponential(p)
	d.summarize(X)
	d.from_summaries()
	assert d.rates.dtype == torch.int32
	assert d._log_rates.dtype == torch.float32

	X = numpy.array(X)
	X = X.astype(numpy.float64)

	p = numpy.array([1.2, 4.1, 0.3], dtype=numpy.float64)
	d = Exponential(p)
	d.summarize(X)
	d.from_summaries()
	assert d.rates.dtype == torch.float64
	assert d._log_rates.dtype == torch.float64


def test_from_summaries_raises():
	assert_raises(AttributeError, Exponential().from_summaries)


def test_fit(X):
	d = Exponential([1.3, 2.3, 6.1])
	d.fit(X[:4])
	assert_array_almost_equal(d.rates, [1.0, 0.8, 0.8])
	assert_array_almost_equal(d._log_rates, [0.0, -0.223144, -0.223144])
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0, 0.0])

	d.fit(X[4:])
	assert_array_almost_equal(d.rates, [0.3, 1.0, 0.75])
	assert_array_almost_equal(d._log_rates, [-1.203973,  0.0, -0.287682])
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0, 0.0])

	d = Exponential([1.3, 2.3, 6.1])
	d.fit(X)
	assert_array_almost_equal(d.rates, [0.5, 0.875, 0.777778])
	assert_array_almost_equal(d._log_rates, [-0.693147, -0.133531, -0.251314])
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0, 0.0])


	d = Exponential()
	d.fit(X[:4])
	assert_array_almost_equal(d.rates, [1.0, 0.8, 0.8])
	assert_array_almost_equal(d._log_rates, [0.0, -0.223144, -0.223144])
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0, 0.0])

	d.fit(X[4:])
	assert_array_almost_equal(d.rates, [0.3, 1.0, 0.75])
	assert_array_almost_equal(d._log_rates, [-1.203973,  0.0, -0.287682])
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0, 0.0])

	d = Exponential()
	d.fit(X)
	assert_array_almost_equal(d.rates, [0.5, 0.875, 0.777778])
	assert_array_almost_equal(d._log_rates, [-0.693147, -0.133531, -0.251314])
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0, 0.0])


def test_fit_weighted(X, w):
	d = Exponential([1.3, 2.3, 6.1])
	d.fit(X[:4], sample_weights=w[:4])
	assert_array_almost_equal(d.rates, [3.0, 1.5, 1.5])
	assert_array_almost_equal(d._log_rates, [1.098612, 0.405465, 0.405465])
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0, 0.0])

	d.fit(X[4:], sample_weights=w[4:])
	assert_array_almost_equal(d.rates, [0.333333, 1. , 2.])
	assert_array_almost_equal(d._log_rates, [-1.098612, 0., 0.693147])
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0, 0.0])

	d = Exponential([1.3, 2.3, 6.1])
	d.fit(X, sample_weights=w)
	assert_array_almost_equal(d.rates, [0.44, 1.1, 1.833333])
	assert_array_almost_equal(d._log_rates, [-0.820981,  0.09531, 0.606136])
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0, 0.0])


	d = Exponential()
	d.fit(X[:4], sample_weights=w[:4])
	assert_array_almost_equal(d.rates, [3.0, 1.5, 1.5])
	assert_array_almost_equal(d._log_rates, [1.098612, 0.405465, 0.405465])
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0, 0.0])

	d.fit(X[4:], sample_weights=w[4:])
	assert_array_almost_equal(d.rates, [0.333333, 1. , 2.])
	assert_array_almost_equal(d._log_rates, [-1.098612, 0., 0.693147])
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0, 0.0])

	d = Exponential()
	d.fit(X, sample_weights=w)
	assert_array_almost_equal(d.rates, [0.44, 1.1, 1.833333])
	assert_array_almost_equal(d._log_rates, [-0.820981,  0.09531, 0.606136])
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0, 0.0])

	X = [[1.2, 0.5, 1.1, 1.9],
	     [6.2, 1.1, 2.4, 1.1]] 

	w = [[1.1], [3.5]]

	d = Exponential()
	d.fit(X, sample_weights=w)
	assert_array_almost_equal(d.rates, [0.199826, 1.045455, 0.478668, 0.774411])
	assert_array_almost_equal(d._log_rates, [-1.610307,  0.044452, -0.736748, 
		-0.255653])
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0, 0.0, 0.0])


def test_fit_chain(X):
	d = Exponential().fit(X[:4])
	assert_array_almost_equal(d.rates, [1.0, 0.8, 0.8])
	assert_array_almost_equal(d._log_rates, [0.0, -0.223144, -0.223144])
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0, 0.0])

	d.fit(X[4:])
	assert_array_almost_equal(d.rates, [0.3, 1.0, 0.75])
	assert_array_almost_equal(d._log_rates, [-1.203973,  0.0, -0.287682])
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0, 0.0])

	d = Exponential().fit(X)
	assert_array_almost_equal(d.rates, [0.5, 0.875, 0.777778])
	assert_array_almost_equal(d._log_rates, [-0.693147, -0.133531, -0.251314])
	assert_array_almost_equal(d._w_sum, [0.0, 0.0, 0.0])
	assert_array_almost_equal(d._xw_sum, [0.0, 0.0, 0.0])


def test_fit_dtypes(X):
	X = numpy.array(X)
	X = X.astype(numpy.float32)

	p = numpy.array([1.2, 4.1, 0.3], dtype=numpy.float32)
	d = Exponential(p).fit(X)
	assert d.rates.dtype == torch.float32
	assert d._log_rates.dtype == torch.float32

	p = numpy.array([1.2, 4.1, 0.3], dtype=numpy.float64)
	d = Exponential(p).fit(X)
	assert d.rates.dtype == torch.float64
	assert d._log_rates.dtype == torch.float64

	p = numpy.array([1, 4, 0], dtype=numpy.int32)
	d = Exponential(p).fit(X)
	assert d.rates.dtype == torch.int32
	assert d._log_rates.dtype == torch.float32

	X = numpy.array(X)
	X = X.astype(numpy.float64)

	p = numpy.array([1.2, 4.1, 0.3], dtype=numpy.float64)
	d = Exponential(p).fit(X)
	assert d.rates.dtype == torch.float64
	assert d._log_rates.dtype == torch.float64


def test_fit_raises(X, w):
	d = Exponential([1.2, 1.8, 2.1])
	assert_raises(ValueError, d.fit, [[1.1, 1.2, 1.9, 1.2]])
	assert_raises(ValueError, d.fit, [[1.1]])
	assert_raises(ValueError, d.fit, [1.1, 1.2, 1.9, 1.2])
	assert_raises(ValueError, d.fit, [1.1])
	assert_raises(ValueError, d.fit, [1.1, 1.2, 1.9])
	assert_raises(ValueError, d.fit, [[[1.1]]])

	d = Exponential([1.2])
	assert_raises(ValueError, d.fit, [[1.1, 1.2, 1.9, 1.2]])
	assert_raises(ValueError, d.fit, [[]])
	assert_raises(ValueError, d.fit, [1.1, 1.2, 1.9, 1.2])
	assert_raises(ValueError, d.fit, [1.1])
	assert_raises(ValueError, d.fit, [[[1.1]]])

	d = Exponential([1.2, 1.8, 2.1])
	assert_raises(ValueError, d.fit, [X])
	assert_raises(ValueError, d.fit, [X], w)
	assert_raises(ValueError, d.fit, X, [w])
	assert_raises(ValueError, d.fit, X, w[:3])
	assert_raises(ValueError, d.fit, X[:3], w)
