# test_bayes_classifier.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy
import torch
import pytest

from torchegranate.hmm import HiddenMarkovModel
from torchegranate._dense_hmm import _DenseHMM
from torchegranate.distributions import Exponential

from .distributions._utils import _test_initialization_raises_one_parameter
from .distributions._utils import _test_initialization
from .distributions._utils import _test_predictions
from .distributions._utils import _test_efd_from_summaries
from .distributions._utils import _test_raises

from nose.tools import assert_raises
from numpy.testing import assert_array_almost_equal


MIN_VALUE = 0
MAX_VALUE = None
VALID_VALUE = 1.2


@pytest.fixture
def X():
	return [[[1, 2, 0],
	      [0, 0, 1],
	      [1, 1, 2],
	      [2, 2, 2],
	      [3, 1, 0]],
	     [[5, 1, 4],
	      [2, 1, 0],
	      [1, 0, 2],
	      [1, 1, 0],
	      [0, 2, 1]]]


@pytest.fixture
def w():
	return [1, 2.3]


@pytest.fixture
def model():
	starts = [0.2, 0.8]
	ends = [0.1, 0.1]

	edges = [[0.1, 0.8],
	         [0.3, 0.6]]

	d = [Exponential([2.1, 0.3, 0.1]), Exponential([1.5, 3.1, 2.2])]
	model = HiddenMarkovModel(nodes=d, edges=edges, starts=starts, ends=ends,
		kind='dense')
	model.bake()
	return model

###


def test_initialization():
	d = [Exponential(), Exponential()]
	model = HiddenMarkovModel(d, kind='dense')

	assert model.inertia == 0.0
	assert model.frozen == False
	assert model.kind == 'dense'

	assert model.n_nodes == 2

	assert_array_almost_equal(model.ends, torch.ones(2) / 2.0)
	assert_array_almost_equal(model.starts, torch.ones(2) / 2.0)
	assert_array_almost_equal(model.edges, torch.ones(2, 2) / 2.0)

	assert_raises(AttributeError, getattr, model._model, "_xw_sum")
	assert_raises(AttributeError, getattr, model._model, "_xw_starts_sum")
	assert_raises(AttributeError, getattr, model._model, "_xw_ends_sum")


def test_initialization_raises():
	d = [Exponential(), Exponential()]

	assert_raises(ValueError, HiddenMarkovModel, d, edges=[0.2, 0.2, 0.6],
		kind='dense')
	assert_raises(ValueError, HiddenMarkovModel, d, edges=[0.2, 1.0],
		kind='dense')
	assert_raises(ValueError, HiddenMarkovModel, d, kind='dense', 
		edges=[[-0.2, 0.9], [0.2, 0.8]])
	assert_raises(ValueError, HiddenMarkovModel, d, kind='dense',
		edges=[[0.3, 1.1], [0.2, 0.8]])
	assert_raises(ValueError, HiddenMarkovModel, d, kind='dense',
		edges=[[0.2, 0.6, 0.2], [0.2, 0.7, 0.1], [0.1, 0.3, 0.6]])
	assert_raises(ValueError, HiddenMarkovModel, d, kind='dense',
		edges=[[[0.2, 0.8], [0.2, 0.8]]])

	assert_raises(ValueError, HiddenMarkovModel, d, kind='dense',
		starts=[0.1, 0.3])
	assert_raises(ValueError, HiddenMarkovModel, d, kind='dense',
		starts=[0.1, 1.2])
	assert_raises(ValueError, HiddenMarkovModel, d, kind='dense',
		starts=[-0.1, 1.1])
	assert_raises(ValueError, HiddenMarkovModel, d, kind='dense',
		starts=[0.5, 0.6])
	assert_raises(ValueError, HiddenMarkovModel, d, kind='dense',
		starts=[0.1, 0.3, 0.3, 0.3])
	assert_raises(ValueError, HiddenMarkovModel, d, kind='dense',
		starts=[[0.1, 0.9]])
	assert_raises(ValueError, HiddenMarkovModel, d, kind='dense',
		starts=[[0.1], [0.9]])

	assert_raises(ValueError, HiddenMarkovModel, d, kind='dense',
		ends=[0.1, 1.2])
	assert_raises(ValueError, HiddenMarkovModel, d, kind='dense',
		ends=[-0.1, 1.1])
	assert_raises(ValueError, HiddenMarkovModel, d, kind='dense',
		ends=[0.1, 0.3, 0.3, 0.3])
	assert_raises(ValueError, HiddenMarkovModel, d, kind='dense',
		ends=[[0.1, 0.9]])
	assert_raises(ValueError, HiddenMarkovModel, d, kind='dense',
		ends=[[0.1], [0.9]])

	assert_raises(ValueError, HiddenMarkovModel, d, kind='dense', max_iter=0)
	assert_raises(ValueError, HiddenMarkovModel, d, kind='dense', max_iter=-1)
	assert_raises(ValueError, HiddenMarkovModel, d, kind='dense', max_iter=1.3)
	
	assert_raises(ValueError, HiddenMarkovModel, d, kind='dense', tol=-1)

	assert_raises((ValueError, TypeError), HiddenMarkovModel, Exponential,
		kind='dense')
	assert_raises(ValueError, HiddenMarkovModel, d, kind='dense', inertia=-0.4)
	assert_raises(ValueError, HiddenMarkovModel, d, kind='dense', inertia=1.2)
	assert_raises(ValueError, HiddenMarkovModel, d, kind='dense', inertia=1.2, 
		frozen="true")
	assert_raises(ValueError, HiddenMarkovModel, d, kind='dense', inertia=1.2, 
		frozen=3)
	

def test_reset_cache(model, X):
	model.summarize(X)
	assert_array_almost_equal(model._model._xw_sum, 
		[[2.666842e-04, 1.895242e+00], 
		 [2.635101e+00, 3.469390e+00]])
	assert_array_almost_equal(model._model._xw_starts_sum, [0.136405, 1.863595])
	assert_array_almost_equal(model._model._xw_ends_sum, [0.876264, 1.123736])

	model._reset_cache()
	assert_array_almost_equal(model._model._xw_sum, [[0., 0.], [0., 0.]])
	assert_array_almost_equal(model._model._xw_starts_sum, [0., 0.])
	assert_array_almost_equal(model._model._xw_ends_sum, [0., 0.])


def test_initialize(X):
	d = [Exponential(), Exponential()]
	model = HiddenMarkovModel(d, kind='dense', random_state=0)

	d1 = model.nodes[0].distribution
	d2 = model.nodes[1].distribution

	assert model.d is None
	assert model.n_nodes == 2
	assert model._initialized == False
	assert model._model is None

	assert d1._initialized == False
	assert d2._initialized == False

	assert_raises(AttributeError, getattr, model._model, "_xw_sum")
	assert_raises(AttributeError, getattr, model._model, "_xw_starts_sum")
	assert_raises(AttributeError, getattr, model._model, "_xw_ends_sum")

	model.bake()
	model._initialize(X)
	assert model._initialized == True
	assert model.d == 3
	assert isinstance(model._model, _DenseHMM)

	assert d1._initialized == True
	assert d2._initialized == True

	assert_array_almost_equal(d1.scales, [1.5, 1. , 2. ])
	assert_array_almost_equal(d2.scales, [1.75, 1.25, 0.  ])


###


def test_emission_matrix(model, X):
	e = model._emission_matrix(X)

	assert_array_almost_equal(e, 
		[[[ -4.3782,  -3.6372],
          [ -7.2354,  -2.7799],
          [-21.0449,  -4.2237],
          [-24.8544,  -5.2129],
          [ -1.9973,  -4.6479]],

         [[-42.9497,  -7.7994],
          [ -1.5211,  -3.9812],
          [-17.7116,  -3.9011],
          [ -1.0449,  -3.3146],
          [-13.9020,  -3.4250]]], 4)


def test_emission_matrix_raises(model, X):
	f = getattr(model, "_emission_matrix")

	assert_raises(ValueError, f, [X])
	assert_raises(ValueError, f, X[0])
	assert_raises((ValueError, TypeError, RuntimeError), f, X[0][0])

	if MIN_VALUE is not None:
		assert_raises(ValueError, f, 
			[[[MIN_VALUE-0.1 for i in range(model.d)] for j in range(4)]])
	
	if MAX_VALUE is not None:
		assert_raises(ValueError, f, 
			[[[MAX_VALUE+0.1 for i in range(model.d)] for j in range(4)]])


def test_log_probability(model, X):
	logp = model.log_probability(X)
	assert_array_almost_equal(logp, [-22.8266, -22.8068], 4)


def test_log_probability_raises(model, X):
	f = getattr(model, "log_probability")

	assert_raises(ValueError, f, [X])
	assert_raises(ValueError, f, X[0])
	assert_raises((ValueError, TypeError, RuntimeError), f, X[0][0])

	if MIN_VALUE is not None:
		assert_raises(ValueError, f, 
			[[[MIN_VALUE-0.1 for i in range(model.d)] for j in range(4)]])
	
	if MAX_VALUE is not None:
		assert_raises(ValueError, f, 
			[[[MAX_VALUE+0.1 for i in range(model.d)] for j in range(4)]])


def test_probability(model, X):
	logp = model.probability(X)
	assert_array_almost_equal(logp, [1.2205e-09, 1.2449e-09], 4)


def test_probability_raises(model, X):
	f = getattr(model, "probability")

	assert_raises(ValueError, f, [X])
	assert_raises(ValueError, f, X[0])
	assert_raises((ValueError, TypeError, RuntimeError), f, X[0][0])

	if MIN_VALUE is not None:
		assert_raises(ValueError, f, 
			[[[MIN_VALUE-0.1 for i in range(model.d)] for j in range(4)]])
	
	if MAX_VALUE is not None:
		assert_raises(ValueError, f, 
			[[[MAX_VALUE+0.1 for i in range(model.d)] for j in range(4)]])


def test_forward(model, X):
	y_hat = model.forward(X)

	assert_array_almost_equal(y_hat,
		[[[ -5.9877,  -3.8603],
          [-12.2607,  -7.0036],
          [-29.2507, -11.7311],
          [-37.7895, -17.4549],
          [-20.6561, -22.6136]],

         [[-44.5591,  -8.0226],
          [-10.7476, -12.5146],
          [-30.3480, -14.7513],
          [-17.0002, -18.5767],
          [-32.7223, -20.5042]]], 4)


def test_forward_raises(model, X):
	f = getattr(model, "forward")

	assert_raises(ValueError, f, [X])
	assert_raises(ValueError, f, X[0])
	assert_raises((ValueError, TypeError, RuntimeError), f, X[0][0])

	if MIN_VALUE is not None:
		assert_raises(ValueError, f, 
			[[[MIN_VALUE-0.1 for i in range(model.d)] for j in range(4)]])
	
	if MAX_VALUE is not None:
		assert_raises(ValueError, f, 
			[[[MAX_VALUE+0.1 for i in range(model.d)] for j in range(4)]])


def test_backward(model, X):
	y_hat = model.backward(X)

	assert_array_almost_equal(y_hat,
		[[[-18.8311, -19.1130],
          [-15.5423, -15.8300],
          [-10.8078, -11.0955],
          [ -6.1547,  -5.3717],
          [ -2.3026,  -2.3026]],

         [[-15.5896, -14.7842],
          [-12.1797, -12.4674],
          [ -8.8158,  -8.0555],
          [ -5.9508,  -6.2384],
          [ -2.3026,  -2.3026]]], 4)


def test_backward_raises(model, X):
	f = getattr(model, "backward")

	assert_raises(ValueError, f, [X])
	assert_raises(ValueError, f, X[0])
	assert_raises((ValueError, TypeError, RuntimeError), f, X[0][0])

	if MIN_VALUE is not None:
		assert_raises(ValueError, f, 
			[[[MIN_VALUE-0.1 for i in range(model.d)] for j in range(4)]])
	
	if MAX_VALUE is not None:
		assert_raises(ValueError, f, 
			[[[MAX_VALUE+0.1 for i in range(model.d)] for j in range(4)]])


def test_forward_backward(model, X):
	expected_transitions, fb, starts, ends, logp = model.forward_backward(X)

	assert_array_almost_equal(expected_transitions,
		[[[2.6353e-04, 1.4304e-01], [8.8289e-01, 2.9738e+00]],
         [[3.1500e-06, 1.7522e+00], [1.7522e+00, 4.9559e-01]]], 3)

	assert_array_almost_equal(fb,
		[[[-1.9921e+00, -1.4665e-01],
          [-4.9764e+00, -6.9228e-03],
          [-1.7232e+01, -3.2831e-08],
          [-2.1118e+01, -6.7415e-10],
          [-1.3209e-01, -2.0896e+00]],

         [[-3.7342e+01,  0.0000e+00],
          [-1.2056e-01, -2.1752e+00],
          [-1.6357e+01, -7.8752e-08],
          [-1.4412e-01, -2.0083e+00],
          [-1.2218e+01, -4.9402e-06]]], 3)

	assert_array_almost_equal(starts, 
		[[1.3641e-01, 8.6359e-01],
         [6.0619e-17, 1.0000e+00]], 3)

	assert_array_almost_equal(ends,
		[[8.7626e-01, 1.2374e-01],
         [4.9402e-06, 1.0000e+00]], 3)

	assert_array_almost_equal(logp, [-22.8266, -22.8068], 3)


def test_predict(model, X):
	y_hat = model.predict(X)
	assert_array_almost_equal(y_hat, 
		[[1, 1, 1, 1, 0],
         [1, 0, 1, 0, 1]], 4)


def test_predict_raises(model, X):
	f = getattr(model, "predict")

	assert_raises(ValueError, f, [X])
	assert_raises(ValueError, f, X[0])
	assert_raises((ValueError, TypeError, RuntimeError), f, X[0][0])

	if MIN_VALUE is not None:
		assert_raises(ValueError, f, 
			[[[MIN_VALUE-0.1 for i in range(model.d)] for j in range(4)]])
	
	if MAX_VALUE is not None:
		assert_raises(ValueError, f, 
			[[[MAX_VALUE+0.1 for i in range(model.d)] for j in range(4)]])


def test_predict_proba(model, X):
	y_hat = model.predict_proba(X)

	assert_array_almost_equal(y_hat,
		[[[1.3641e-01, 8.6359e-01],
          [6.8989e-03, 9.9310e-01],
          [3.2831e-08, 1.0000e+00],
          [6.7415e-10, 1.0000e+00],
          [8.7626e-01, 1.2374e-01]],

         [[6.0619e-17, 1.0000e+00],
          [8.8642e-01, 1.1358e-01],
          [7.8752e-08, 1.0000e+00],
          [8.6578e-01, 1.3422e-01],
          [4.9402e-06, 1.0000e+00]]], 4)

	assert_array_almost_equal(torch.sum(y_hat, dim=-1),
		[[1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
         [1.0000, 1.0000, 1.0000, 1.0000, 1.0000]])


def test_predict_proba_raises(model, X):
	f = getattr(model, "predict_proba")

	assert_raises(ValueError, f, [X])
	assert_raises(ValueError, f, X[0])
	assert_raises((ValueError, TypeError, RuntimeError), f, X[0][0])

	if MIN_VALUE is not None:
		assert_raises(ValueError, f, 
			[[[MIN_VALUE-0.1 for i in range(model.d)] for j in range(4)]])
	
	if MAX_VALUE is not None:
		assert_raises(ValueError, f, 
			[[[MAX_VALUE+0.1 for i in range(model.d)] for j in range(4)]])


def test_predict_log_proba(model, X):
	y_hat = model.predict_log_proba(X)

	assert_array_almost_equal(y_hat,
		[[[-1.9921e+00, -1.4665e-01],
          [-4.9764e+00, -6.9228e-03],
          [-1.7232e+01, -3.2831e-08],
          [-2.1118e+01, -6.7415e-10],
          [-1.3209e-01, -2.0896e+00]],

         [[-3.7342e+01,  0.0000e+00],
          [-1.2056e-01, -2.1752e+00],
          [-1.6357e+01, -7.8752e-08],
          [-1.4412e-01, -2.0083e+00],
          [-1.2218e+01, -4.9402e-06]]], 3)


def test_predict_log_proba_raises(model, X):
	f = getattr(model, "predict_log_proba")

	assert_raises(ValueError, f, [X])
	assert_raises(ValueError, f, X[0])
	assert_raises((ValueError, TypeError, RuntimeError), f, X[0][0])

	if MIN_VALUE is not None:
		assert_raises(ValueError, f, 
			[[[MIN_VALUE-0.1 for i in range(model.d)] for j in range(4)]])
	
	if MAX_VALUE is not None:
		assert_raises(ValueError, f, 
			[[[MAX_VALUE+0.1 for i in range(model.d)] for j in range(4)]])
###


def test_partial_summarize(model, X):
	d1 = model.nodes[0].distribution
	d2 = model.nodes[1].distribution
	model.summarize(X[:1])

	assert_array_almost_equal(model._model._xw_sum,
		[[2.635342e-04, 1.430404e-01], [8.828947e-01, 2.973801e+00]])
	assert_array_almost_equal(model._model._xw_starts_sum, [0.136405, 0.863595])
	assert_array_almost_equal(model._model._xw_ends_sum, [0.876259, 0.123741])

	assert_array_almost_equal(d1._w_sum, [1.019563, 1.019563, 1.019563])
	assert_array_almost_equal(d1._xw_sum, [2.765183, 1.149069, 0.006899])

	assert_array_almost_equal(d2._w_sum, [3.980437, 3.980437, 3.980437])
	assert_array_almost_equal(d2._xw_sum, [4.234817, 4.85093 , 4.993101])	

	model.summarize(X[1:])
	assert_array_almost_equal(model._model._xw_sum, 
		[[2.666842e-04, 1.895242e+00], [2.635101e+00, 3.469390e+00]])
	assert_array_almost_equal(model._model._xw_starts_sum, [0.136405, 1.863595])
	assert_array_almost_equal(model._model._xw_ends_sum, [0.876264, 1.123736])

	assert_array_almost_equal(d1._w_sum, [2.771773, 2.771773, 2.771773])
	assert_array_almost_equal(d1._xw_sum, [5.403807, 2.901284, 0.006904])

	assert_array_almost_equal(d2._w_sum, [7.228228, 7.228228, 7.228228])
	assert_array_almost_equal(d2._xw_sum, [10.596193,  8.098716, 11.993096])	


def test_summarize(model, X):
	d1 = model.nodes[0].distribution
	d2 = model.nodes[1].distribution
	model.summarize(X)

	assert_array_almost_equal(model._model._xw_sum, 
		[[2.666842e-04, 1.895242e+00], [2.635101e+00, 3.469390e+00]])
	assert_array_almost_equal(model._model._xw_starts_sum, [0.136405, 1.863595])
	assert_array_almost_equal(model._model._xw_ends_sum, [0.876264, 1.123736])

	assert_array_almost_equal(d1._w_sum, [2.771773, 2.771773, 2.771773])
	assert_array_almost_equal(d1._xw_sum, [5.403807, 2.901284, 0.006904])

	assert_array_almost_equal(d2._w_sum, [7.228228, 7.228228, 7.228228])
	assert_array_almost_equal(d2._xw_sum, [10.596193,  8.098716, 11.993096])	


def test_summarize_weighted(model, X, w):
	d1 = model.nodes[0].distribution
	d2 = model.nodes[1].distribution
	model.summarize(X, sample_weight=w)

	assert_array_almost_equal(model._model._xw_sum, 
		[[2.707793e-04, 4.173104e+00], [4.912969e+00, 4.113656e+00]])
	assert_array_almost_equal(model._model._xw_starts_sum, [0.136405, 3.163595])
	assert_array_almost_equal(model._model._xw_ends_sum, [0.876271, 2.423729])

	assert_array_almost_equal(d1._w_sum, [5.049645, 5.049645, 5.049645])
	assert_array_almost_equal(d1._xw_sum, [8.834019e+00, 5.179163e+00, 
		6.910709e-03])

	assert_array_almost_equal(d2._w_sum, [11.450356, 11.450356, 11.450356])
	assert_array_almost_equal(d2._xw_sum, [18.86598 , 12.320837, 21.09309])	


def test_summarize_raises(model, X, w):
	assert_raises(ValueError, model.summarize, [X])
	assert_raises(ValueError, model.summarize, X[0])
	assert_raises((ValueError, TypeError), model.summarize, X[0][0])
	assert_raises(ValueError, model.summarize, 
		[[-0.1 for i in range(3)] for x in X])

	assert_raises(ValueError, model.summarize, [X], w)
	assert_raises(ValueError, model.summarize, X, [w])
	assert_raises(ValueError, model.summarize, [X], [w])
	assert_raises(ValueError, model.summarize, X[:len(X)-1], w)
	assert_raises(ValueError, model.summarize, X, w[:len(w)-1])


def test_from_summaries(model, X):
	d1 = model.nodes[0].distribution
	d2 = model.nodes[1].distribution

	model.summarize(X)
	model.from_summaries()

	assert_array_almost_equal(model.starts, [-2.685274, -0.07064])
	assert_array_almost_equal(model.ends, [-1.151575, -1.861335])
	assert_array_almost_equal(model.edges, 
		[[-9.248932, -0.380141], [-1.009072, -0.734015]])

	assert_array_almost_equal(d1.scales, [1.949585, 1.046725, 0.002491])
	assert_array_almost_equal(d1._w_sum, [0., 0., 0.])
	assert_array_almost_equal(d1._xw_sum, [0., 0., 0.])

	assert_array_almost_equal(d2.scales, [1.465946, 1.120429, 1.659203])
	assert_array_almost_equal(d2._w_sum, [0., 0., 0.])
	assert_array_almost_equal(d2._xw_sum, [0., 0., 0.])


def test_from_summaries_weighted(model, X, w):
	d1 = model.nodes[0].distribution
	d2 = model.nodes[1].distribution

	model.summarize(X, sample_weight=w)
	model.from_summaries()

	assert_array_almost_equal(model.starts, [-3.186049, -0.042213])
	assert_array_almost_equal(model.ends, [-1.751398, -1.552713])
	assert_array_almost_equal(model.edges, 
		[[-9.833524, -0.190658], [-0.846142, -1.023709]])

	assert_array_almost_equal(d1.scales, 
		[1.749434e+00, 1.025649e+00, 1.368553e-03])
	assert_array_almost_equal(d1._w_sum, [0., 0., 0.])
	assert_array_almost_equal(d1._xw_sum, [0., 0., 0.])

	assert_array_almost_equal(d2.scales, [1.647633, 1.076022, 1.842134])
	assert_array_almost_equal(d2._w_sum, [0., 0., 0.])
	assert_array_almost_equal(d2._xw_sum, [0., 0., 0.])


def test_from_summaries_inertia(X):
	d = [Exponential([2.1, 0.3, 0.1]), Exponential([1.5, 3.1, 2.2])]
	model = HiddenMarkovModel(nodes=d, edges=[[0.1, 0.8], [0.3, 0.6]], 
		starts=[0.2, 0.8], ends=[0.1, 0.1], kind='dense', inertia=0.3)
	model.bake()

	d1 = model.nodes[0].distribution
	d2 = model.nodes[1].distribution

	model.summarize(X)
	model.from_summaries()

	assert_array_almost_equal(model.starts, [-2.362523, -0.116391])
	assert_array_almost_equal(model.ends, [-1.496878, -1.99371])
	assert_array_almost_equal(model.edges, 
		[[-7.165028, -0.333042], [-1.067542, -0.667058]])

	assert_array_almost_equal(d1.scales, [1.949585, 1.046725, 0.002491])
	assert_array_almost_equal(d1._w_sum, [0., 0., 0.])
	assert_array_almost_equal(d1._xw_sum, [0., 0., 0.])

	assert_array_almost_equal(d2.scales, [1.465946, 1.120429, 1.659203])
	assert_array_almost_equal(d2._w_sum, [0., 0., 0.])
	assert_array_almost_equal(d2._xw_sum, [0., 0., 0.])


	d = [Exponential([2.1, 0.3, 0.1], inertia=0.25), 
	     Exponential([1.5, 3.1, 2.2], inertia=0.83)]
	model = HiddenMarkovModel(nodes=d, edges=[[0.1, 0.8], [0.3, 0.6]], 
		starts=[0.2, 0.8], ends=[0.1, 0.1], kind='dense', inertia=0.0)
	model.bake()

	d1 = model.nodes[0].distribution
	d2 = model.nodes[1].distribution

	model.summarize(X)
	model.from_summaries()

	assert_array_almost_equal(model.starts, [-2.685274, -0.07064])
	assert_array_almost_equal(model.ends, [-1.151575, -1.861335])
	assert_array_almost_equal(model.edges, 
		[[-9.248932, -0.380141], [-1.009072, -0.734015]])

	assert_array_almost_equal(d1.scales, [1.987189, 0.860044, 0.026868])
	assert_array_almost_equal(d1._w_sum, [0., 0., 0.])
	assert_array_almost_equal(d1._xw_sum, [0., 0., 0.])

	assert_array_almost_equal(d2.scales, [1.494211, 2.763473, 2.108064])
	assert_array_almost_equal(d2._w_sum, [0., 0., 0.])
	assert_array_almost_equal(d2._xw_sum, [0., 0., 0.])


def test_from_summaries_weighted_inertia(X, w):
	d = [Exponential([2.1, 0.3, 0.1]), Exponential([1.5, 3.1, 2.2])]
	model = HiddenMarkovModel(nodes=d, edges=[[0.1, 0.8], [0.3, 0.6]], 
		starts=[0.2, 0.8], ends=[0.1, 0.1], kind='dense', inertia=0.3)
	model.bake()

	d1 = model.nodes[0].distribution
	d2 = model.nodes[1].distribution

	model.summarize(X, sample_weight=w)
	model.from_summaries()

	assert_array_almost_equal(model.starts, [-2.713066, -0.096492])
	assert_array_almost_equal(model.ends, [-1.916754, -1.777675])
	assert_array_almost_equal(model.edges, 
		[[-7.574243, -0.200404], [-0.953491, -0.869844]])

	assert_array_almost_equal(d1.scales, 
		[1.749434e+00, 1.025649e+00, 1.368553e-03], 3)
	assert_array_almost_equal(d1._w_sum, [0., 0., 0.])
	assert_array_almost_equal(d1._xw_sum, [0., 0., 0.])

	assert_array_almost_equal(d2.scales, [1.647633, 1.076022, 1.842134])
	assert_array_almost_equal(d2._w_sum, [0., 0., 0.])
	assert_array_almost_equal(d2._xw_sum, [0., 0., 0.])


	d = [Exponential([2.1, 0.3, 0.1], inertia=0.25), 
	     Exponential([1.5, 3.1, 2.2], inertia=0.83)]
	model = HiddenMarkovModel(nodes=d, edges=[[0.1, 0.8], [0.3, 0.6]], 
		starts=[0.2, 0.8], ends=[0.1, 0.1], kind='dense', inertia=0.0)
	model.bake()

	d1 = model.nodes[0].distribution
	d2 = model.nodes[1].distribution

	model.summarize(X, sample_weight=w)
	model.from_summaries()

	assert_array_almost_equal(model.starts, [-3.186049, -0.042213])
	assert_array_almost_equal(model.ends, [-1.751398, -1.552713])
	assert_array_almost_equal(model.edges, 
		[[-9.833524, -0.190658], [-0.846142, -1.023709]])

	assert_array_almost_equal(d1.scales, [1.837075, 0.844237, 0.026026])
	assert_array_almost_equal(d1._w_sum, [0., 0., 0.])
	assert_array_almost_equal(d1._xw_sum, [0., 0., 0.])

	assert_array_almost_equal(d2.scales, [1.525098, 2.755924, 2.139163])
	assert_array_almost_equal(d2._w_sum, [0., 0., 0.])
	assert_array_almost_equal(d2._xw_sum, [0., 0., 0.])


def test_from_summaries_frozen(model, X):
	d = [Exponential([2.1, 0.3, 0.1]), Exponential([1.5, 3.1, 2.2])]
	model = HiddenMarkovModel(nodes=d, edges=[[0.1, 0.8], [0.3, 0.6]], 
		starts=[0.2, 0.8], ends=[0.1, 0.1], kind='dense', frozen=True)
	model.bake()

	d1 = model.nodes[0].distribution
	d2 = model.nodes[1].distribution

	model.summarize(X)
	model.from_summaries()

	assert_array_almost_equal(model.starts, [-1.609438, -0.223144])
	assert_array_almost_equal(model.ends, [-2.302585, -2.302585])
	assert_array_almost_equal(model.edges, 
		[[-2.302585, -0.223144], [-1.203973, -0.510826]])

	assert_array_almost_equal(d1.scales, [1.949585, 1.046725, 0.002491])
	assert_array_almost_equal(d1._w_sum, [0., 0., 0.])
	assert_array_almost_equal(d1._xw_sum, [0., 0., 0.])

	assert_array_almost_equal(d2.scales, [1.465946, 1.120429, 1.659203])
	assert_array_almost_equal(d2._w_sum, [0., 0., 0.])
	assert_array_almost_equal(d2._xw_sum, [0., 0., 0.])


	d = [Exponential([2.1, 0.3, 0.1], frozen=True), 
	     Exponential([1.5, 3.1, 2.2], frozen=True)]
	model = HiddenMarkovModel(nodes=d, edges=[[0.1, 0.8], [0.3, 0.6]], 
		starts=[0.2, 0.8], ends=[0.1, 0.1], kind='dense', inertia=0.0)
	model.bake()

	d1 = model.nodes[0].distribution
	d2 = model.nodes[1].distribution

	model.summarize(X)
	model.from_summaries()

	assert_array_almost_equal(model.starts, [-2.685274, -0.07064])
	assert_array_almost_equal(model.ends, [-1.151575, -1.861335])
	assert_array_almost_equal(model.edges, 
		[[-9.248932, -0.380141], [-1.009072, -0.734015]])

	assert_array_almost_equal(d1.scales, [2.1, 0.3, 0.1])
	assert_array_almost_equal(d1._w_sum, [0., 0., 0.])
	assert_array_almost_equal(d1._xw_sum, [0., 0., 0.])

	assert_array_almost_equal(d2.scales, [1.5, 3.1, 2.2])
	assert_array_almost_equal(d2._w_sum, [0., 0., 0.])
	assert_array_almost_equal(d2._xw_sum, [0., 0., 0.])


def test_fit(X):
	X = torch.tensor(numpy.array(X) + 1)

	d = [Exponential([2.1, 0.3, 0.1]), Exponential([1.5, 3.1, 2.2])]
	model = HiddenMarkovModel(nodes=d, edges=[[0.1, 0.8], [0.3, 0.6]], 
		starts=[0.2, 0.8], ends=[0.1, 0.1], kind='dense', max_iter=1)
	model.bake()
	model.fit(X)
	
	d1 = model.nodes[0].distribution
	d2 = model.nodes[1].distribution

	assert_array_almost_equal(model.starts, [-1.489857e+01, -3.385568e-07], 4)
	assert_array_almost_equal(model.ends, [-1.110725, -1.609444])
	assert_array_almost_equal(model.edges, 
		[[-23.442373,  -0.399463], [-11.607553,  -0.223153]])

	assert_array_almost_equal(d1.scales, [3.021216, 2.007029, 1.000361])
	assert_array_almost_equal(d1._w_sum, [0., 0., 0.])
	assert_array_almost_equal(d1._xw_sum, [0., 0., 0.])

	assert_array_almost_equal(d2.scales, [2.599996, 2.100001, 2.200011])
	assert_array_almost_equal(d2._w_sum, [0., 0., 0.])
	assert_array_almost_equal(d2._xw_sum, [0., 0., 0.])

	d = [Exponential([2.1, 0.3, 0.1]), Exponential([1.5, 3.1, 2.2])]
	model = HiddenMarkovModel(nodes=d, edges=[[0.1, 0.8], [0.3, 0.6]], 
		starts=[0.2, 0.8], ends=[0.1, 0.1], kind='dense', max_iter=5)
	model.bake()
	model.fit(X)

	
	d1 = model.nodes[0].distribution
	d2 = model.nodes[1].distribution

	assert_array_almost_equal(model.starts, [-1.545504e+01, -1.940718e-07], 4)
	assert_array_almost_equal(model.ends, [-0.758036, -1.609449])
	assert_array_almost_equal(model.edges, 
		[[-23.906063,  -0.632214], [-11.732584,  -0.223151]])

	assert_array_almost_equal(d1.scales, [2.603264, 2.076076, 1.532971])
	assert_array_almost_equal(d1._w_sum, [0., 0., 0.])
	assert_array_almost_equal(d1._xw_sum, [0., 0., 0.])

	assert_array_almost_equal(d2.scales, [2.6     , 2.1     , 2.200005])
	assert_array_almost_equal(d2._w_sum, [0., 0., 0.])
	assert_array_almost_equal(d2._xw_sum, [0., 0., 0.])


def test_fit_weighted(X, w):
	X = torch.tensor(numpy.array(X) + 1)

	d = [Exponential([2.1, 0.3, 0.1]), Exponential([1.5, 3.1, 2.2])]
	model = HiddenMarkovModel(nodes=d, edges=[[0.1, 0.8], [0.3, 0.6]], 
		starts=[0.2, 0.8], ends=[0.1, 0.1], kind='dense', max_iter=1)
	model.bake()
	model.fit(X, sample_weight=w)
	
	d1 = model.nodes[0].distribution
	d2 = model.nodes[1].distribution

	assert_array_almost_equal(model.starts, [-1.5399e+01, -2.0519e-07], 3)
	assert_array_almost_equal(model.ends, [-1.732272, -1.609437])
	assert_array_almost_equal(model.edges, 
		[[-23.970318,  -0.194656], [-11.483337,  -0.223157]], 5)

	assert_array_almost_equal(d1.scales, [2.801925, 2.003776, 1.000194])
	assert_array_almost_equal(d1._w_sum, [0., 0., 0.])
	assert_array_almost_equal(d1._xw_sum, [0., 0., 0.])

	assert_array_almost_equal(d2.scales, [2.678787, 2.060607, 2.278801])
	assert_array_almost_equal(d2._w_sum, [0., 0., 0.])
	assert_array_almost_equal(d2._xw_sum, [0., 0., 0.])

	d = [Exponential([2.1, 0.3, 0.1]), Exponential([1.5, 3.1, 2.2])]
	model = HiddenMarkovModel(nodes=d, edges=[[0.1, 0.8], [0.3, 0.6]], 
		starts=[0.2, 0.8], ends=[0.1, 0.1], kind='dense', max_iter=5)
	model.bake()
	model.fit(X, sample_weight=w)

	
	d1 = model.nodes[0].distribution
	d2 = model.nodes[1].distribution

	assert_array_almost_equal(model.starts, [-1.6093e+01, -1.0250e-07], 3)
	assert_array_almost_equal(model.ends, [-1.469704, -1.609439])
	assert_array_almost_equal(model.edges, 
		[[-24.481024,  -0.261356], [-11.632328,  -0.223154]], 5)

	assert_array_almost_equal(d1.scales, [2.324057, 2.012569, 1.522347])
	assert_array_almost_equal(d1._w_sum, [0., 0., 0.])
	assert_array_almost_equal(d1._xw_sum, [0., 0., 0.])

	assert_array_almost_equal(d2.scales, [2.678791, 2.060607, 2.278795])
	assert_array_almost_equal(d2._w_sum, [0., 0., 0.])
	assert_array_almost_equal(d2._xw_sum, [0., 0., 0.])


def test_fit_raises(model, X, w):
	assert_raises(ValueError, model.fit, [X])
	assert_raises(ValueError, model.fit, X[0])
	assert_raises((ValueError, TypeError), model.fit, X[0][0])
	assert_raises(ValueError, model.fit, 
		[[-0.1 for i in range(3)] for x in X])

	assert_raises(ValueError, model.fit, [X], w)
	assert_raises(ValueError, model.fit, X, [w])
	assert_raises(ValueError, model.fit, [X], [w])
	assert_raises(ValueError, model.fit, X[:len(X)-1], w)
	assert_raises(ValueError, model.fit, X, w[:len(w)-1])
