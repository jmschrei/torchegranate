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
from numpy.testing import assert_array_equal
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
def priors():
	return torch.tensor(numpy.array([
		[0.5, 0.5],
	    [0.5, 0.5],
	    [0.5, 0.5],
	    [0.0, 1.0],
	    [0.5, 0.5],
	    [0.3, 0.7],
	    [0.6, 0.4],
	    [0.5, 0.5],
	    [0.0, 1.0],
	    [1.0, 0.0],
	    [0.5, 0.5]
	]))


@pytest.fixture
def gmm():
	d = [Exponential([2.1, 0.3, 0.1]), Exponential([1.5, 3.1, 2.2])]
	return GeneralMixtureModel(d, priors=[0.7, 0.3])


@pytest.fixture
def hmm():
	d = [Exponential([2.1, 0.3, 0.1]), Exponential([1.5, 3.1, 2.2])]
	return DenseHMM(d, edges=[[0.8, 0.2], [0.4, 0.6]], starts=[0.4, 0.6])


###


def _test_raises(func, X, priors):
	assert_raises(ValueError, func, X, priors+1)
	assert_raises(ValueError, func, X, priors-1)
	assert_raises(ValueError, func, X, priors/2.0)
	assert_raises(ValueError, func, X, priors[:5])
	assert_raises(ValueError, func, X[:5], priors)
	assert_raises(ValueError, func, X, priors[:,0])
	assert_raises(ValueError, func, X, priors[:, :1])


def test_gmm_emission_matrix(gmm, X, priors):
	y_hat = gmm._emission_matrix(X)
	assert_array_almost_equal(y_hat, [
		[ -4.7349,  -4.8411],
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

	y_hat = gmm._emission_matrix(X, priors=priors)
	assert_array_almost_equal(y_hat, [
		[ -5.4281,  -5.5343],
        [ -8.2852,  -4.6770],
        [-22.0947,  -6.1208],
        [    -inf,  -6.4169],
        [ -3.0471,  -6.5450],
        [-44.5103,  -9.3601],
        [ -2.3886,  -6.1015],
        [-18.7614,  -5.7982],
        [    -inf,  -4.5185],
        [-14.2587,     -inf],
        [  1.7148,  -4.2224]], 4)


def test_gmm_emission_matrix_raises(gmm, X, priors):
	_test_raises(gmm._emission_matrix, X, priors)


def test_gmm_probability(gmm, X, priors):
	y_hat = gmm.probability(X)
	assert_array_almost_equal(y_hat, numpy.exp([-4.0935, -3.9571, -5.4276, 
		-6.4169, -2.3241, -9.0034, -1.8418, -5.1051, -1.3582, -4.6289,  
		2.4106]), 3)

	y_hat = gmm.probability(X, priors=priors)
	assert_array_almost_equal(y_hat, [8.3407e-03, 9.5592e-03, 2.1967e-03, 
		1.6337e-03, 4.8933e-02, 8.6094e-05, 9.3998e-02, 3.0330e-03, 
		1.0905e-02, 6.4197e-07, 5.5702e+00], 3)	


def test_gmm_probability_raises(gmm, X, priors):
	_test_raises(gmm.probability, X, priors)


def test_gmm_log_probability(gmm, X, priors):
	y_hat = gmm.log_probability(X)
	assert_array_almost_equal(y_hat, [-4.0935, -3.9571, -5.4276, -6.4169, 
		-2.3241, -9.0034, -1.8418, -5.1051, -1.3582, -4.6289,  2.4106], 4)

	y_hat = gmm.log_probability(X, priors=priors)
	assert_array_almost_equal(y_hat, [-4.7866, -4.6503, -6.1208, -6.4169,
		-3.0173, -9.3601, -2.3645, -5.7982, -4.5185, -14.2587, 1.7174], 4)	


def test_gmm_log_probability_raises(gmm, X, priors):
	_test_raises(gmm.log_probability, X, priors)


def test_gmm_predict(gmm, X, priors):
	y_hat = gmm.predict(X)
	assert_array_equal(y_hat, [0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0])

	y_hat = gmm.predict(X, priors=priors)
	assert_array_equal(y_hat, [0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0])	


def test_gmm_predict_raises(gmm, X, priors):
	_test_raises(gmm.predict, X, priors)


def test_gmm_predict_proba(gmm, X, priors):
	y_hat = gmm.predict_proba(X)
	assert_array_almost_equal(y_hat, [
		[5.2653e-01, 4.7347e-01],
        [2.6385e-02, 9.7361e-01],
        [1.1551e-07, 1.0000e+00],
        [6.8830e-09, 1.0000e+00],
        [9.7063e-01, 2.9372e-02],
        [1.2660e-15, 1.0000e+00],
        [9.6468e-01, 3.5317e-02],
        [2.3451e-06, 1.0000e+00],
        [9.5759e-01, 4.2413e-02],
        [6.5741e-05, 9.9993e-01],
        [9.9737e-01, 2.6323e-03]], 4)

	y_hat = gmm.predict_proba(X, priors=priors)
	assert_array_almost_equal(y_hat, [
		[5.2653e-01, 4.7347e-01],
        [2.6385e-02, 9.7361e-01],
        [1.1551e-07, 1.0000e+00],
        [6.8830e-09, 1.0000e+00],
        [9.7063e-01, 2.9372e-02],
        [1.2660e-15, 1.0000e+00],
        [9.6468e-01, 3.5317e-02],
        [2.3451e-06, 1.0000e+00],
        [9.5759e-01, 4.2413e-02],
        [6.5741e-05, 9.9993e-01],
        [9.9737e-01, 2.6323e-03]], 4)	


def test_gmm_predict_raises(gmm, X, priors):
	_test_raises(gmm.predict_proba, X, priors)


def test_gmm_predict_log_proba(gmm, X, priors):
	y_hat = gmm.predict_log_proba(X)
	assert_array_equal(y_hat, [
		[-6.4145e-01, -7.4766e-01],
        [-3.6350e+00, -2.6740e-02],
        [-1.5974e+01,  0.0000e+00],
        [-1.8794e+01,  0.0000e+00],
        [-2.9812e-02, -3.5277e+00],
        [-3.4303e+01,  0.0000e+00],
        [-3.5955e-02, -3.3434e+00],
        [-1.2963e+01, -2.3842e-06],
        [-4.3338e-02, -3.1603e+00],
        [-9.6298e+00, -6.5804e-05],
        [-2.6357e-03, -5.9399e+00]], 4)

	y_hat = gmm.predict_log_proba(X, priors=priors)
	assert_array_equal(y_hat, [
		[5.2653e-01, 4.7347e-01],
        [2.6385e-02, 9.7361e-01],
        [1.1551e-07, 1.0000e+00],
        [6.8830e-09, 1.0000e+00],
        [9.7063e-01, 2.9372e-02],
        [1.2660e-15, 1.0000e+00],
        [9.6468e-01, 3.5317e-02],
        [2.3451e-06, 1.0000e+00],
        [9.5759e-01, 4.2413e-02],
        [6.5741e-05, 9.9993e-01],
        [9.9737e-01, 2.6323e-03]])	


def test_gmm_predict_raises(gmm, X, priors):
	_test_raises(gmm.predict_log_proba, X, priors)