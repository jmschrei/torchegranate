# test_ConditionalCategorical.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy
import torch
import pytest

from torchegranate.distributions import ConditionalCategorical

from ._utils import _test_initialization_raises_one_parameter
from ._utils import _test_initialization
from ._utils import _test_predictions
from ._utils import _test_raises

from nose.tools import assert_raises
from numpy.testing import assert_array_almost_equal


MIN_VALUE = 0
MAX_VALUE = 2
VALID_VALUE = numpy.array([1, 2, 0])


@pytest.fixture
def X():
	return [[1, 2, 0],
		 [1, 2, 1],
		 [1, 2, 0],
		 [1, 2, 0],
		 [1, 1, 0],
		 [1, 1, 1],
		 [0, 1, 0]]


@pytest.fixture
def w():
	return [[1.1], [2.8], [0], [0], [5.5], [1.8], [2.3]]



@pytest.fixture
def probs():
	return [[[0.25, 0.75],
			 [0.32, 0.68],
			 [0.5, 0.5]],

			[[0.1, 0.9],
			 [0.3, 0.7],
			 [0.24, 0.76]]]


###


def test_initialization():
	d = ConditionalCategorical()
	_test_initialization(d, None, "probs", 0.0, False, None)
	assert_raises(AttributeError, getattr, d, "_w_sum")
	assert_raises(AttributeError, getattr, d, "_xw_sum")
	assert_raises(AttributeError, getattr, d, "_log_probs")


def test_initialization_float(probs):
	funcs = (lambda x: x, tuple, 
		lambda x: numpy.array(x, dtype=numpy.float32), 
		lambda x: torch.tensor(x, dtype=torch.float32), 
		lambda x: torch.nn.Parameter(torch.tensor(x), requires_grad=False))

	for func in funcs:
		y = func(probs)
		_test_initialization(ConditionalCategorical(y, inertia=0.0, frozen=False), 
			y, "probs", 0.0, False, torch.float32)
		_test_initialization(ConditionalCategorical(y, inertia=0.3, frozen=False), 
			y, "probs", 0.3, False, torch.float32)
		_test_initialization(ConditionalCategorical(y, inertia=1.0, frozen=True), 
			y, "probs", 1.0, True, torch.float32)
		_test_initialization(ConditionalCategorical(y, inertia=1.0, frozen=False), 
			y, "probs", 1.0, False, torch.float32)

	x = numpy.array(probs, dtype=numpy.float64)
	_test_initialization(ConditionalCategorical(x, inertia=0.0, frozen=False), 
		x, "probs", 0.0, False, torch.float64)
	_test_initialization(ConditionalCategorical(x, inertia=0.3, frozen=False), 
		x, "probs", 0.3, False, torch.float64)
	_test_initialization(ConditionalCategorical(x, inertia=1.0, frozen=True), 
		x, "probs", 1.0, True, torch.float64)
	_test_initialization(ConditionalCategorical(x, inertia=1.0, frozen=False), 
		x, "probs", 1.0, False, torch.float64)


def test_initialization_raises(probs):	
	assert_raises(TypeError, ConditionalCategorical, 0.3)
	assert_raises(ValueError, ConditionalCategorical, probs, inertia=-0.4)
	assert_raises(ValueError, ConditionalCategorical, probs, inertia=1.2)
	assert_raises(ValueError, ConditionalCategorical, probs, inertia=1.2, 
		frozen="true")
	assert_raises(ValueError, ConditionalCategorical, probs, inertia=1.2, 
		frozen=3)
	
	assert_raises(ValueError, ConditionalCategorical, inertia=-0.4)
	assert_raises(ValueError, ConditionalCategorical, inertia=1.2)
	assert_raises(ValueError, ConditionalCategorical, inertia=1.2, frozen="true")
	assert_raises(ValueError, ConditionalCategorical, inertia=1.2, frozen=3)

	#assert_raises(ValueError, ConditionalCategorical, numpy.array(probs)+0.001) FIXME
	#assert_raises(ValueError, ConditionalCategorical, numpy.array(probs)-0.001) FIXME

	p = numpy.array(probs)
	p[0, 0] = -0.03
	assert_raises(ValueError, ConditionalCategorical, p)

	p = numpy.array(probs)
	p[0, 0] = 1.03
	assert_raises(ValueError, ConditionalCategorical, p)


def test_reset_cache(X):
	d = ConditionalCategorical()
	d.summarize(X)
	assert_array_almost_equal(d._w_sum, 
		[[ 0.,  1.,  0.], 
		 [ 0.,  2.,  4.]])
	assert_array_almost_equal(d._xw_sum, 
		[[[0., 0.],
          [1., 0.],
          [0., 0.]],

         [[0., 0.],
          [1., 1.],
          [3., 1.]]])

	d._reset_cache()
	assert_array_almost_equal(d._w_sum, 
		[[ 0.,  0.,  0.], 
		 [ 0.,  0.,  0.]])
	assert_array_almost_equal(d._xw_sum,
		[[[0., 0.],
		  [0., 0.],
		  [0., 0.]],

		 [[0., 0.],
		  [0., 0.],
		  [0., 0.]]])

	d = ConditionalCategorical()
	assert_raises(AttributeError, getattr, d, "_w_sum")
	assert_raises(AttributeError, getattr, d, "_xw_sum")
	assert_raises(AttributeError, getattr, d, "_log_probs")

	d._reset_cache()
	assert_raises(AttributeError, getattr, d, "_w_sum")
	assert_raises(AttributeError, getattr, d, "_xw_sum")
	assert_raises(AttributeError, getattr, d, "_log_probs")


def test_initialize(X, probs):
	d = ConditionalCategorical()
	assert d.d is None
	assert d.probs is None
	assert d._initialized == False
	assert_raises(AttributeError, getattr, d, "_w_sum")
	assert_raises(AttributeError, getattr, d, "_xw_sum")
	assert_raises(AttributeError, getattr, d, "_log_probs")

	d._initialize(3, (2, 3, 2))
	assert d._initialized == True
	assert d.probs.shape == (2, 3, 2)
	assert d.d == 3
	assert_array_almost_equal(d.probs,
		[[[0., 0.],
		  [0., 0.],
		  [0., 0.]],

		 [[0., 0.],
		  [0., 0.],
		  [0., 0.]]])
	assert_array_almost_equal(d._w_sum,
		[[ 0.,  0.,  0.], 
		 [ 0.,  0.,  0.]])
	assert_array_almost_equal(d._xw_sum,
		[[[0., 0.],
		  [0., 0.],
		  [0., 0.]],

		 [[0., 0.],
		  [0., 0.],
		  [0., 0.]]])	

	d._initialize(2, (2, 2))
	assert d._initialized == True
	assert d.probs.shape == (2, 2)
	assert d.d == 2
	assert_array_almost_equal(d.probs,
		[[0., 0.],
		 [0., 0.]])
	assert_array_almost_equal(d._w_sum, 
		[0.0, 0.0])
	assert_array_almost_equal(d._xw_sum,
		[[0., 0.],
		 [0., 0.]])	

	d = ConditionalCategorical(probs)
	assert d._initialized == True
	assert d.d == 3
	assert d.n_categories == (2, 3, 2)

	d._initialize(3, (2, 2, 4))
	assert d._initialized == True
	assert d.probs.shape == (2, 2, 4)
	assert d.d == 3
	assert_array_almost_equal(d.probs,
		[[[0., 0., 0., 0.],
		  [0., 0., 0., 0.]],

		 [[0., 0., 0., 0.],
		  [0., 0., 0., 0.]]])

	d = ConditionalCategorical()
	d.summarize(X)
	assert d._initialized == True
	assert d.probs.shape == (2, 3, 2)
	assert d.d == 3
	assert_array_almost_equal(d._w_sum,
		[[ 0.,  1.,  0.], 
		 [ 0.,  2.,  4.]])
	assert_array_almost_equal(d._xw_sum,
		[[[0., 0.],
		  [1., 0.],
		  [0., 0.]],

		 [[0., 0.],
		  [1., 1.],
		  [3., 1.]]])	

	d = ConditionalCategorical()
	d.summarize(X)
	d._initialize(4, (2, 2, 2, 2))
	assert d._initialized == True
	assert d.probs.shape == (2, 2, 2, 2)
	assert d.d == 4
	assert_array_almost_equal(d._w_sum,
		[[[0., 0.],
	      [0., 0.]],

	     [[0., 0.],
	      [0., 0.]]])
	assert_array_almost_equal(d._xw_sum,
		[[[[0., 0.],
		   [0., 0.]],

		  [[0., 0.],
		   [0., 0.]]],


		 [[[0., 0.],
		   [0., 0.]],

		  [[0., 0.],
		   [0., 0.]]]])	


###


def test_probability(X, probs):
	y = [0.24, 0.76, 0.24, 0.24, 0.3 , 0.7 , 0.32]

	d1 = ConditionalCategorical(probs)
	d2 = ConditionalCategorical(numpy.array(probs, dtype=numpy.float64))

	_test_predictions(X, y, d1.probability(X), torch.float32)
	_test_predictions(X, y, d2.probability(X), torch.float64)


def test_probability_dtypes(X, probs):	
	y = ConditionalCategorical(probs).probability(X)
	assert y.dtype == torch.float32

	p = numpy.array(probs, dtype=numpy.float64)
	y = ConditionalCategorical(p).probability(X)
	assert y.dtype == torch.float64


def test_probability_raises(X, probs):
	_test_raises(ConditionalCategorical(probs), "probability", X, min_value=MIN_VALUE, 
		max_value=MAX_VALUE)

	_test_raises(ConditionalCategorical(probs), "probability", X, 
		min_value=MIN_VALUE, max_value=MAX_VALUE)


def test_log_probability(X, probs):
	y = [-1.427116, -0.274437, -1.427116, -1.427116, -1.203973, -0.356675,
           -1.139434]

	d1 = ConditionalCategorical(probs)
	d2 = ConditionalCategorical(numpy.array(probs, dtype=numpy.float64))

	_test_predictions(X, y, d1.log_probability(X), torch.float32)
	_test_predictions(X, y, d2.log_probability(X), torch.float64)


def test_log_probability_dtypes(X, probs):
	y = ConditionalCategorical(probs).log_probability(X)
	assert y.dtype == torch.float32

	p = numpy.array(probs, dtype=numpy.float64)
	y = ConditionalCategorical(p).log_probability(X)
	assert y.dtype == torch.float64


def test_log_probability_raises(X, probs):
	_test_raises(ConditionalCategorical(probs), "log_probability", X, 
		min_value=MIN_VALUE, max_value=MAX_VALUE)

	_test_raises(ConditionalCategorical(probs), "log_probability", X, 
		min_value=MIN_VALUE, max_value=MAX_VALUE)


###


def test_summarize(X, probs):
	d = ConditionalCategorical(probs)
	d.summarize(X[:4])
	assert_array_almost_equal(d._w_sum,
		[[0., 0., 0.],
         [0., 0., 4.]])
	assert_array_almost_equal(d._xw_sum,
		[[[0., 0.],
          [0., 0.],
          [0., 0.]],

         [[0., 0.],
          [0., 0.],
          [3., 1.]]])

	d.summarize(X[4:])
	assert_array_almost_equal(d._w_sum,
		[[0., 1., 0.],
         [0., 2., 4.]])
	assert_array_almost_equal(d._xw_sum,
		[[[0., 0.],
          [1., 0.],
          [0., 0.]],

         [[0., 0.],
          [1., 1.],
          [3., 1.]]])

	d = ConditionalCategorical(probs)
	d.summarize(X)
	assert_array_almost_equal(d._w_sum,
		[[0., 1., 0.],
         [0., 2., 4.]])
	assert_array_almost_equal(d._xw_sum,
		[[[0., 0.],
          [1., 0.],
          [0., 0.]],

         [[0., 0.],
          [1., 1.],
          [3., 1.]]])


	d = ConditionalCategorical()
	d.summarize(X[:4])
	assert_array_almost_equal(d._w_sum,
		[[0., 0., 0.],
         [0., 0., 4.]])
	assert_array_almost_equal(d._xw_sum,
		[[[0., 0.],
          [0., 0.],
          [0., 0.]],

         [[0., 0.],
          [0., 0.],
          [3., 1.]]])

	d.summarize(X[4:])
	assert_array_almost_equal(d._w_sum,
		[[0., 1., 0.],
         [0., 2., 4.]])
	assert_array_almost_equal(d._xw_sum,
		[[[0., 0.],
          [1., 0.],
          [0., 0.]],

         [[0., 0.],
          [1., 1.],
          [3., 1.]]])

	d = ConditionalCategorical()
	d.summarize(X)
	assert_array_almost_equal(d._w_sum,
		[[0., 1., 0.],
         [0., 2., 4.]])
	assert_array_almost_equal(d._xw_sum,
		[[[0., 0.],
          [1., 0.],
          [0., 0.]],

         [[0., 0.],
          [1., 1.],
          [3., 1.]]])


def test_summarize_weighted(X, w, probs):
	d = ConditionalCategorical(probs)
	d.summarize(X[:4], sample_weight=w[:4])
	assert_array_almost_equal(d._w_sum,
		[[0., 0., 0.],
         [0., 0., 3.9]])
	assert_array_almost_equal(d._xw_sum,
		[[[0., 0.],
          [0., 0.],
          [0., 0.]],

         [[0., 0.],
          [0., 0.],
          [1.1, 2.8]]])


	d.summarize(X[4:], sample_weight=w[4:])
	assert_array_almost_equal(d._w_sum,
		[[0., 2.3, 0.],
         [0., 7.3, 3.9]])
	assert_array_almost_equal(d._xw_sum,
		[[[0., 0.],
          [2.3, 0.],
          [0., 0.]],

         [[0., 0.],
          [5.5, 1.8],
          [1.1, 2.8]]])

	d = ConditionalCategorical(probs)
	d.summarize(X, sample_weight=w)
	assert_array_almost_equal(d._w_sum,
		[[0., 2.3, 0.],
         [0., 7.3, 3.9]])
	assert_array_almost_equal(d._xw_sum,
		[[[0., 0.],
          [2.3, 0.],
          [0., 0.]],

         [[0., 0.],
          [5.5, 1.8],
          [1.1, 2.8]]])


def test_summarize_weighted_flat(X, w, probs):
	w = numpy.array(w)[:,0] 

	d = ConditionalCategorical(probs)
	d.summarize(X[:4], sample_weight=w[:4])
	assert_array_almost_equal(d._w_sum,
		[[0., 0., 0.],
         [0., 0., 3.9]])
	assert_array_almost_equal(d._xw_sum,
		[[[0., 0.],
          [0., 0.],
          [0., 0.]],

         [[0., 0.],
          [0., 0.],
          [1.1, 2.8]]])


	d.summarize(X[4:], sample_weight=w[4:])
	assert_array_almost_equal(d._w_sum,
		[[0., 2.3, 0.],
         [0., 7.3, 3.9]])
	assert_array_almost_equal(d._xw_sum,
		[[[0., 0.],
          [2.3, 0.],
          [0., 0.]],

         [[0., 0.],
          [5.5, 1.8],
          [1.1, 2.8]]])

	d = ConditionalCategorical(probs)
	d.summarize(X, sample_weight=w)
	assert_array_almost_equal(d._w_sum,
		[[0., 2.3, 0.],
         [0., 7.3, 3.9]])
	assert_array_almost_equal(d._xw_sum,
		[[[0., 0.],
          [2.3, 0.],
          [0., 0.]],

         [[0., 0.],
          [5.5, 1.8],
          [1.1, 2.8]]])


def test_summarize_dtypes(X, w, probs):
	X = numpy.array(X)
	probs = numpy.array(probs, dtype=numpy.float32)

	X = X.astype(numpy.int32)
	d = ConditionalCategorical(probs)
	assert d._xw_sum.dtype == torch.float32
	d.summarize(X)
	assert d._xw_sum.dtype == torch.float32

	X = X.astype(numpy.int64)
	d = ConditionalCategorical(probs)
	assert d._xw_sum.dtype == torch.float32
	d.summarize(X)
	assert d._xw_sum.dtype == torch.float32

	X = X.astype(numpy.int32)
	d = ConditionalCategorical(probs.astype(numpy.float64))
	assert d._xw_sum.dtype == torch.float64
	d.summarize(X)
	assert d._xw_sum.dtype == torch.float64

	X = X.astype(numpy.int64)
	d = ConditionalCategorical(probs.astype(numpy.float64))
	assert d._xw_sum.dtype == torch.float64
	d.summarize(X)
	assert d._xw_sum.dtype == torch.float64


def test_summarize_raises(X, w, probs):
	_test_raises(ConditionalCategorical(probs), "summarize", X, w=w, 
		min_value=MIN_VALUE, max_value=MAX_VALUE)

	_test_raises(ConditionalCategorical(), "summarize", X, w=w, 
		min_value=MIN_VALUE, max_value=MAX_VALUE)


def _test_efd_from_summaries(d, name1, name2, values):
	assert_array_almost_equal(getattr(d, name1), values, 4)
	assert_array_almost_equal(getattr(d, name2), numpy.log(values), 2)
	assert_array_almost_equal(d._w_sum, numpy.zeros(d.probs.shape[:-1]))
	assert_array_almost_equal(d._xw_sum, numpy.zeros(d.probs.shape))


def test_from_summaries(X, probs):
	d = ConditionalCategorical(probs)
	d.summarize(X)
	d.from_summaries()
	assert_raises(AssertionError, assert_array_almost_equal, probs, d.probs)

	for param in probs, None:
		d = ConditionalCategorical(param)
		d.summarize(X[:4])
		d.from_summaries()
		_test_efd_from_summaries(d, "probs", "_log_probs",
			[[[0.5000, 0.5000],
	          [0.5000, 0.5000],
	          [0.5000, 0.5000]],

	         [[0.5000, 0.5000],
	          [0.5000, 0.5000],
	          [0.7500, 0.2500]]])

		d.summarize(X[4:])
		d.from_summaries()
		_test_efd_from_summaries(d, "probs", "_log_probs", 
			[[[0.5000, 0.5000],
 	          [1.0000, 0.0000],
	          [0.5000, 0.5000]],

	         [[0.5000, 0.5000],
	          [0.5000, 0.5000],
	          [0.5000, 0.5000]]])

		d = ConditionalCategorical(param)
		d.summarize(X[:4])
		d.summarize(X[4:])
		d.from_summaries()
		_test_efd_from_summaries(d, "probs", "_log_probs", 
			[[[0.5000, 0.5000],
         	  [1.0000, 0.0000],
         	  [0.5000, 0.5000]],

        	 [[0.5000, 0.5000],
         	  [0.5000, 0.5000],
         	  [0.7500, 0.2500]]])

		d = ConditionalCategorical(param)
		d.summarize(X)
		d.from_summaries()
		_test_efd_from_summaries(d, "probs", "_log_probs", 
			[[[0.5000, 0.5000],
         	  [1.0000, 0.0000],
         	  [0.5000, 0.5000]],

        	 [[0.5000, 0.5000],
         	  [0.5000, 0.5000],
         	  [0.7500, 0.2500]]])


def test_from_summaries_weighted(X, w, probs):
	for param in probs, None:
		d = ConditionalCategorical(probs)
		d.summarize(X[:4], sample_weight=w[:4])
		d.from_summaries()
		_test_efd_from_summaries(d, "probs", "_log_probs", 
			[[[0.5000, 0.5000],
	          [0.5000, 0.5000],
	          [0.5000, 0.5000]],

	         [[0.5000, 0.5000],
	          [0.5000, 0.5000],
	          [0.2821, 0.7179]]])

		d.summarize(X[4:], sample_weight=w[4:])
		d.from_summaries()
		_test_efd_from_summaries(d, "probs", "_log_probs", 
			[[[0.5000, 0.5000],
       		  [1.0000, 0.0000],
         	  [0.5000, 0.5000]],

        	 [[0.5000, 0.5000],
         	  [0.7534, 0.2466],
         	  [0.5000, 0.5000]]])

		d = ConditionalCategorical(probs)
		d.summarize(X, sample_weight=w)
		d.from_summaries()
		_test_efd_from_summaries(d, "probs", "_log_probs", 
			[[[0.5000, 0.5000],
       		  [1.0000, 0.0000],
         	  [0.5000, 0.5000]],

        	 [[0.5000, 0.5000],
         	  [0.7534, 0.2466],
         	  [0.2821, 0.7179]]])


def test_from_summaries_null(probs):
	d = ConditionalCategorical(probs)
	d.from_summaries()
	assert_raises(AssertionError, assert_array_almost_equal, d.probs, probs)
	assert_array_almost_equal(d._w_sum, 
		[[0.0, 0.0, 0.0],
		 [0.0, 0.0, 0.0]])
	assert_array_almost_equal(d._xw_sum, numpy.zeros_like(probs))

	d = ConditionalCategorical(probs, inertia=0.5)
	d.from_summaries()
	assert_raises(AssertionError, assert_array_almost_equal, d.probs, probs)
	assert_array_almost_equal(d._w_sum,
		[[0.0, 0.0, 0.0],
		 [0.0, 0.0, 0.0]])
	assert_array_almost_equal(d._xw_sum, numpy.zeros_like(probs))

	d = ConditionalCategorical(probs, inertia=0.5, frozen=True)
	d.from_summaries()
	_test_efd_from_summaries(d, "probs", "_log_probs", probs)


def test_from_summaries_inertia(X, w, probs):
	d = ConditionalCategorical(probs, inertia=0.3)
	d.summarize(X[:4])
	d.from_summaries()
	_test_efd_from_summaries(d, "probs", "_log_probs", 
		[[[0.4250, 0.5750],
          [0.4460, 0.5540],
          [0.5000, 0.5000]],

         [[0.3800, 0.6200],
          [0.4400, 0.5600],
          [0.5970, 0.4030]]])

	d.summarize(X[4:])
	d.from_summaries()
	_test_efd_from_summaries(d, "probs", "_log_probs", 
		[[[0.4775, 0.5225],
          [0.8338, 0.1662],
          [0.5000, 0.5000]],

         [[0.4640, 0.5360],
          [0.4820, 0.5180],
          [0.5291, 0.4709]]])

	d = ConditionalCategorical(probs, inertia=0.3)
	d.summarize(X)
	d.from_summaries()
	_test_efd_from_summaries(d, "probs", "_log_probs", 
		[[[0.4250, 0.5750],
          [0.7960, 0.2040],
          [0.5000, 0.5000]],

         [[0.3800, 0.6200],
          [0.4400, 0.5600],
          [0.5970, 0.4030]]])


def test_from_summaries_weighted_inertia(X, w, probs):
	d = ConditionalCategorical(probs, inertia=0.3)
	d.summarize(X, sample_weight=w)
	d.from_summaries()
	_test_efd_from_summaries(d, "probs", "_log_probs", 
		[[[0.4250, 0.5750],
          [0.7960, 0.2040],
          [0.5000, 0.5000]],

         [[0.3800, 0.6200],
          [0.6174, 0.3826],
          [0.2694, 0.7306]]])

	d = ConditionalCategorical(probs, inertia=1.0)
	d.summarize(X[:4])
	d.from_summaries()
	_test_efd_from_summaries(d, "probs", "_log_probs", probs)

	d.summarize(X[4:])
	d.from_summaries()
	_test_efd_from_summaries(d, "probs", "_log_probs", probs)

	d = ConditionalCategorical(probs, inertia=1.0)
	d.summarize(X)
	d.from_summaries()
	_test_efd_from_summaries(d, "probs", "_log_probs", probs)


def test_from_summaries_frozen(X, w, probs):
	d = ConditionalCategorical(probs, frozen=True)
	d.summarize(X[:4])
	assert_array_almost_equal(d._w_sum, 
		[[0.0, 0.0, 0.0],
		 [0.0, 0.0, 0.0]])
	assert_array_almost_equal(d._xw_sum, numpy.zeros_like(probs))

	d.from_summaries()
	_test_efd_from_summaries(d, "probs", "_log_probs", probs)

	d.summarize(X[4:])
	assert_array_almost_equal(d._w_sum, 
		[[0.0, 0.0, 0.0],
		 [0.0, 0.0, 0.0]])
	assert_array_almost_equal(d._xw_sum, numpy.zeros_like(probs))

	d.from_summaries()
	_test_efd_from_summaries(d, "probs", "_log_probs", probs)

	d = ConditionalCategorical(probs, frozen=True)
	d.summarize(X)
	assert_array_almost_equal(d._w_sum, 
		[[0.0, 0.0, 0.0],
		 [0.0, 0.0, 0.0]])
	assert_array_almost_equal(d._xw_sum, numpy.zeros_like(probs))

	d.from_summaries()
	_test_efd_from_summaries(d, "probs", "_log_probs", probs)

	d = ConditionalCategorical(probs, frozen=True)
	d.summarize(X, sample_weight=w)
	assert_array_almost_equal(d._w_sum, 
		[[0.0, 0.0, 0.0],
		 [0.0, 0.0, 0.0]])
	assert_array_almost_equal(d._xw_sum, numpy.zeros_like(probs))

	d.from_summaries()
	_test_efd_from_summaries(d, "probs", "_log_probs", probs)


def test_from_summaries_dtypes(X, probs):
	p = numpy.array(probs, dtype=numpy.float32)
	d = ConditionalCategorical(p)
	d.summarize(X)
	d.from_summaries()
	assert d.probs.dtype == torch.float32
	assert d._log_probs.dtype == torch.float32

	p = numpy.array(probs, dtype=numpy.float64)
	d = ConditionalCategorical(p)
	d.summarize(X)
	d.from_summaries()
	assert d.probs.dtype == torch.float64
	assert d._log_probs.dtype == torch.float64


def test_from_summaries_raises():
	assert_raises(AttributeError, ConditionalCategorical().from_summaries)


def test_fit(X, w, probs):
	d = ConditionalCategorical(probs)
	d.fit(X)
	assert_raises(AssertionError, assert_array_almost_equal, probs, d.probs)

	for param in probs, None:
		d = ConditionalCategorical(param)
		d.fit(X[:4])
		_test_efd_from_summaries(d, "probs", "_log_probs",
			[[[0.5000, 0.5000],
	          [0.5000, 0.5000],
	          [0.5000, 0.5000]],

	         [[0.5000, 0.5000],
	          [0.5000, 0.5000],
	          [0.7500, 0.2500]]])

		d.fit(X[4:])
		_test_efd_from_summaries(d, "probs", "_log_probs", 
			[[[0.5000, 0.5000],
 	          [1.0000, 0.0000],
	          [0.5000, 0.5000]],

	         [[0.5000, 0.5000],
	          [0.5000, 0.5000],
	          [0.5000, 0.5000]]])

		d = ConditionalCategorical(param)
		d.fit(X)
		_test_efd_from_summaries(d, "probs", "_log_probs", 
			[[[0.5000, 0.5000],
         	  [1.0000, 0.0000],
         	  [0.5000, 0.5000]],

        	 [[0.5000, 0.5000],
         	  [0.5000, 0.5000],
         	  [0.7500, 0.2500]]])


def test_fit_weighted(X, w, probs):
	for param in probs, None:
		d = ConditionalCategorical(probs)
		d.fit(X[:4], sample_weight=w[:4])
		_test_efd_from_summaries(d, "probs", "_log_probs", 
			[[[0.5000, 0.5000],
	          [0.5000, 0.5000],
	          [0.5000, 0.5000]],

	         [[0.5000, 0.5000],
	          [0.5000, 0.5000],
	          [0.2821, 0.7179]]])

		d.fit(X[4:], sample_weight=w[4:])
		_test_efd_from_summaries(d, "probs", "_log_probs", 
			[[[0.5000, 0.5000],
       		  [1.0000, 0.0000],
         	  [0.5000, 0.5000]],

        	 [[0.5000, 0.5000],
         	  [0.7534, 0.2466],
         	  [0.5000, 0.5000]]])

		d = ConditionalCategorical(probs)
		d.fit(X, sample_weight=w)
		_test_efd_from_summaries(d, "probs", "_log_probs", 
			[[[0.5000, 0.5000],
       		  [1.0000, 0.0000],
         	  [0.5000, 0.5000]],

        	 [[0.5000, 0.5000],
         	  [0.7534, 0.2466],
         	  [0.2821, 0.7179]]])


def test_fit_chain(X):
	d = ConditionalCategorical().fit(X[:4])
	_test_efd_from_summaries(d, "probs", "_log_probs",
		[[[0.5000, 0.5000],
          [0.5000, 0.5000],
          [0.5000, 0.5000]],

         [[0.5000, 0.5000],
          [0.5000, 0.5000],
          [0.7500, 0.2500]]])

	d.fit(X[4:])
	_test_efd_from_summaries(d, "probs", "_log_probs", 
		[[[0.5000, 0.5000],
	          [1.0000, 0.0000],
          [0.5000, 0.5000]],

         [[0.5000, 0.5000],
          [0.5000, 0.5000],
          [0.5000, 0.5000]]])


	d = ConditionalCategorical().fit(X)
	_test_efd_from_summaries(d, "probs", "_log_probs", 
		[[[0.5000, 0.5000],
     	  [1.0000, 0.0000],
     	  [0.5000, 0.5000]],

    	 [[0.5000, 0.5000],
     	  [0.5000, 0.5000],
     	  [0.7500, 0.2500]]])


def test_fit_dtypes(X, probs):
	p = numpy.array(probs, dtype=numpy.float32)
	d = ConditionalCategorical(p)
	d.fit(X)
	assert d.probs.dtype == torch.float32
	assert d._log_probs.dtype == torch.float32

	p = numpy.array(probs, dtype=numpy.float64)
	d = ConditionalCategorical(p)
	d.fit(X)
	assert d.probs.dtype == torch.float64
	assert d._log_probs.dtype == torch.float64


def test_fit_raises(X, w, probs):
	_test_raises(ConditionalCategorical(probs), "fit", X, w=w, min_value=MIN_VALUE, 
		max_value=MAX_VALUE)

	_test_raises(ConditionalCategorical(), "fit", X, w=w, min_value=MIN_VALUE, 
		max_value=MAX_VALUE)
