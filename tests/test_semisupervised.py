# test_bayes_classifier.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import os
import numpy
import torch
import pytest

from torchegranate.gmm import GeneralMixtureModel
from torchegranate.distributions import Exponential

from .distributions._utils import _test_initialization_raises_one_parameter
from .distributions._utils import _test_initialization
from .distributions._utils import _test_predictions
from .distributions._utils import _test_efd_from_summaries
from .distributions._utils import _test_raises

from nose.tools import assert_raises
from numpy.testing import assert_array_almost_equal


inf = float("inf")


@pytest.fixture
def X():
	return [[1, 2, 0],
	     [0, 0, 1],
	     [1, 1, 2],
	     [2, 2, 2],
	     [3, 1, 0],
	     [5, 1, 4],
	     [2, 1, 0],
	     [1, 0, 2],
	     [1, 1, 0],
	     [0, 2, 1],
	     [0, 0, 0]]


@pytest.fixture
def X_masked(X):
	mask = torch.tensor(numpy.array([
		[False, True,  True ],
		[True,  True,  False],
		[False, False, False],
		[True,  True,  True ],
		[False, True,  False],
		[True,  True,  True ],
		[False, False, False],
		[True,  False, True ],
		[True,  True,  True ],
		[True,  True,  True ],
		[True,  False, True ]]))

	X = torch.tensor(numpy.array(X))
	return torch.masked.MaskedTensor(X, mask=mask)


@pytest.fixture
def w():
	return [[1], [2], [0], [0], [5], [1], [2], [1], [1], [2], [0]]


@pytest.fixture
def y():
	y_ = torch.tensor([0, 0, -1, 1, -1, -1, -1, -1, 1, -1, -1])
	return torch.masked.MaskedTensor(y_, mask=y_ != -1)


@pytest.fixture
def model():
	d = [Exponential([2.1, 0.3, 0.1]), Exponential([1.5, 3.1, 2.2])]
	return GeneralMixtureModel(d, priors=[0.7, 0.3])


###


def test_emission_matrix(model, X, y):
	e = model._emission_matrix(X)
	assert_array_almost_equal(e, 
		[[ -4.7349,  -4.8411],
         [ -7.5921,  -3.9838],
         [-21.4016,  -5.4276],
         [-25.2111,  -6.4169],
         [ -2.3540,  -5.8519],
         [-43.3063,  -9.0034],
         [ -1.8778,  -5.1852],
         [-18.0682,  -5.1051],
         [ -1.4016,  -4.5185],
         [-14.2587,  -4.6290],
         [  2.4079,  -3.5293]], 4)

	ey = model._emission_matrix(X, y=y)
	assert_array_almost_equal(ey, 
		[[       0,     -inf],
         [       0,     -inf],
         [-21.4016,  -5.4276],
         [    -inf,        0],
         [ -2.3540,  -5.8519],
         [-43.3063,  -9.0034],
         [ -1.8778,  -5.1852],
         [-18.0682,  -5.1051],
         [    -inf,        0],
         [-14.2587,  -4.6290],
         [  2.4079,  -3.5293]], 4)


def test_emission_matrix_raises(model, X, y):
	assert_raises(ValueError, model._emission_matrix, X, y+1)
	assert_raises(ValueError, model._emission_matrix, X, y-1)
	assert_raises(ValueError, model._emission_matrix, X, y[:5])
	assert_raises(ValueError, model._emission_matrix, X[:5], y)

	y = torch.randint(2, size=(len(X), 3))
	mask = torch.randint(2, size=(len(X), 3)).type(torch.bool)
	y = torch.masked.MaskedTensor(y, mask=mask)
	assert_raises(ValueError, model._emission_matrix, X, y)


def test_partial_summarize(model, X, y):
	model.summarize(X[:4], y=y[:4])
	assert_array_almost_equal(model._w_sum, [2, 2])
	assert_array_almost_equal(model.distributions[0]._w_sum, 
		[2, 2, 2])
	assert_array_almost_equal(model.distributions[0]._xw_sum, 
		[1, 2, 1])
	assert_array_almost_equal(model.distributions[1]._w_sum, 
		[2, 2, 2])
	assert_array_almost_equal(model.distributions[1]._xw_sum, 
		[3, 3, 4])

	model.summarize(X[4:], y=y[4:])
	assert_array_almost_equal(model._w_sum, [4.932748, 6.067252])
	assert_array_almost_equal(model.distributions[0]._w_sum, 
		[4.932748, 4.932748, 4.932748])
	assert_array_almost_equal(model.distributions[0]._xw_sum, 
		[5.841254, 3.935443, 1.000071])
	assert_array_almost_equal(model.distributions[1]._w_sum, 
		[6.067252, 6.067252, 6.067252])
	assert_array_almost_equal(model.distributions[1]._xw_sum, 
		[10.158746,  7.064557, 10.999929])


def test_full_summarize(model, X, y):
	model.summarize(X)
	assert_array_almost_equal(model._w_sum, [4.443249, 6.556751])
	assert_array_almost_equal(model.distributions[0]._w_sum, 
		[4.443249, 4.443249, 4.443249])
	assert_array_almost_equal(model.distributions[0]._xw_sum, 
		[6.32537 , 3.946088, 0.026456])
	assert_array_almost_equal(model.distributions[1]._w_sum, 
		[6.556752, 6.556752, 6.556752])
	assert_array_almost_equal(model.distributions[1]._xw_sum, 
		[9.674629,  7.053912, 11.973544])

	d = [Exponential([2.1, 0.3, 0.1]), Exponential([1.5, 3.1, 2.2])]
	model = GeneralMixtureModel(d, priors=[0.7, 0.3])
	model.summarize(X, y=y)
	assert_array_almost_equal(model._w_sum, [4.932748, 6.067252])
	assert_array_almost_equal(model.distributions[0]._w_sum, 
		[4.932748, 4.932748, 4.932748])
	assert_array_almost_equal(model.distributions[0]._xw_sum, 
		[5.841254, 3.935443, 1.000071])
	assert_array_almost_equal(model.distributions[1]._w_sum, 
		[6.067252, 6.067252, 6.067252])
	assert_array_almost_equal(model.distributions[1]._xw_sum, 
		[10.158746,  7.064557, 10.999929])


def test_summarize_raises(model, X, y):
	assert_raises(ValueError, model.summarize, X, None, y+1)
	assert_raises(ValueError, model.summarize, X, None, y-1)
	assert_raises(ValueError, model.summarize, X, None, y[:5])
	assert_raises(ValueError, model.summarize, X[:5], None, y)

	y = torch.randint(2, size=(len(X), 3))
	mask = torch.randint(2, size=(len(X), 3)).type(torch.bool)
	y = torch.masked.MaskedTensor(y, mask=mask)
	assert_raises(ValueError, model.summarize, X, None, y)


def test_fit(X, y):
	d = [Exponential([2.1, 0.3, 1.1]), Exponential([1.5, 3.1, 2.2])]
	model = GeneralMixtureModel(d, max_iter=1)
	model.fit(X, y=y)

	assert_array_almost_equal(model._w_sum, [0., 0.])
	assert_array_almost_equal(model.priors, [0.491916, 0.508084])
	assert_array_almost_equal(model._log_priors, numpy.log(
		[0.491916, 0.508084]))


	d = [Exponential([2.1, 0.3, 1.1]), Exponential([1.5, 3.1, 2.2])]
	model = GeneralMixtureModel(d, max_iter=5)
	model.fit(X, y=y)

	assert_array_almost_equal(model._w_sum, [0., 0.])
	assert_array_almost_equal(model.priors, [0.470019, 0.529981])
	assert_array_almost_equal(model._log_priors, numpy.log(
		[0.470019, 0.529981]))


def test_fit_weighted(X, w, y):
	d = [Exponential([2.1, 0.3, 1.1]), Exponential([1.5, 3.1, 2.2])]
	model = GeneralMixtureModel(d, max_iter=1)
	model.fit(X, sample_weight=w, y=y)

	assert_array_almost_equal(model._w_sum, [0., 0.])
	assert_array_almost_equal(model.priors, [0.533468, 0.466532])
	assert_array_almost_equal(model._log_priors, numpy.log(
		[0.533468, 0.466532]))


	d = [Exponential([2.1, 0.3, 1.1]), Exponential([1.5, 3.1, 2.2])]
	model = GeneralMixtureModel(d, max_iter=5)
	model.fit(X, sample_weight=w, y=y)

	assert_array_almost_equal(model._w_sum, [0., 0.])
	assert_array_almost_equal(model.priors, [0.665037, 0.334963])
	assert_array_almost_equal(model._log_priors, numpy.log(
		[0.665037, 0.334963]))


def test_masked_emission_matrix(model, X, X_masked, y):
	X = torch.tensor(numpy.array(X))
	mask = torch.ones_like(X).type(torch.bool)
	X_ = torch.masked.MaskedTensor(X, mask=mask)
	e = model._emission_matrix(X_, y=y)
	assert_array_almost_equal(e, 
		[[       0,     -inf],
         [       0,     -inf],
         [-21.4016,  -5.4276],
         [    -inf,        0],
         [ -2.3540,  -5.8519],
         [-43.3063,  -9.0034],
         [ -1.8778,  -5.1852],
         [-18.0682,  -5.1051],
         [    -inf,        0],
         [-14.2587,  -4.6290],
         [  2.4079,  -3.5293]], 4)

	d = [Exponential([2.1, 0.3, 0.1]), Exponential([1.5, 3.1, 2.2])]
	model = GeneralMixtureModel(d)
	e = model._emission_matrix(X_masked, y=y)
	assert_array_almost_equal(e, 
		[[  0.0000,     -inf],
         [  0.0000,     -inf],
         [ -0.6931,  -0.6931],
         [    -inf,   0.0000],
         [ -2.8225,  -2.1471],
         [-43.6428,  -8.4926],
         [ -0.6931,  -0.6931],
         [-19.6087,  -3.4628],
         [    -inf,   0.0000],
         [-14.5952,  -4.1182],
         [  0.8675,  -1.8871]], 4)
	

def test_masked_summarize(model, X, X_masked, w, y):
	X = torch.tensor(numpy.array(X))
	mask = torch.ones_like(X).type(torch.bool)
	X_ = torch.masked.MaskedTensor(X, mask=mask)

	d = [Exponential([2.1, 0.3, 0.1]), Exponential([1.5, 3.1, 2.2])]
	model = GeneralMixtureModel(d, priors=[0.7, 0.3])
	model.summarize(X_, sample_weight=w, y=y)
	assert_array_almost_equal(model._w_sum, [9.782642, 5.217357])
	assert_array_almost_equal(model.distributions[0]._w_sum, 
		[9.782642, 9.782642, 9.782642])
	assert_array_almost_equal(model.distributions[0]._xw_sum, 
		[19.418161,  8.782771,  2.000136])
	assert_array_almost_equal(model.distributions[1]._w_sum, 
		[5.217357, 5.217357, 5.217357])
	assert_array_almost_equal(model.distributions[1]._xw_sum, 
		[7.581837, 6.217227, 7.999864])

	d = [Exponential([2.1, 0.3, 0.1]), Exponential([1.5, 3.1, 2.2])]
	model = GeneralMixtureModel(d, priors=[0.7, 0.3])
	model.summarize(X_masked, sample_weight=w, y=y)
	assert_array_almost_equal(model._w_sum, [5.714504, 7.285496])
	assert_array_almost_equal(model.distributions[0]._w_sum, 
		[2.000132, 5.714504, 1.000132])
	assert_array_almost_equal(model.distributions[0]._xw_sum, 
		[2.269437e-07, 4.714635e+00, 1.319366e-04])
	assert_array_almost_equal(model.distributions[1]._w_sum, 
		[4.999868, 6.285496, 4.999868])
	assert_array_almost_equal(model.distributions[1]._xw_sum, 
		[7.      , 8.285364, 7.999868])


def test_masked_fit(model, X_masked, y):
	d = [Exponential([2.1, 1.5, 1.0]), Exponential([1.5, 3.1, 2.2])]
	model = GeneralMixtureModel(d, max_iter=5)
	model.fit(X_masked, y=y)

	assert_array_almost_equal(model._w_sum, [0., 0.])
	assert_array_almost_equal(model.priors, [0.499671, 0.500329])
	assert_array_almost_equal(model._log_priors, 
		numpy.log([0.499671, 0.500329]))
