# test_utils.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy
import torch

from torchegranate.distributions._utils import _cast_as_tensor
from torchegranate.distributions._utils import _update_parameter
from torchegranate.distributions._utils import _check_parameter

from nose.tools import assert_almost_equal
from nose.tools import assert_equal
from nose.tools import assert_raises
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal


def _test_cast(y, dtype, ndim):
	x = _cast_as_tensor(y)
	assert isinstance(x, torch.Tensor)
	assert x.dtype == dtype
	assert x.ndim == ndim

	if type(x) not in (numpy.ndarray, torch.Tensor) and ndim == 0:
		if type(x) in float:
			assert_almost_equal(x, y)
		else:
			assert_equal(x, y)
	else:
		assert_array_almost_equal(x, y)


def test_cast_as_tensor_bool():
	_test_cast(False, torch.bool, 0)
	_test_cast(True, torch.bool, 0)


def test_cast_as_tensor_int():
	_test_cast(5, torch.int64, 0)
	_test_cast(0, torch.int64, 0)
	_test_cast(-1, torch.int64, 0)


def test_cast_as_tensor_float():
	_test_cast(1.2, torch.float32, 0)
	_test_cast(0.0, torch.float32, 0)
	_test_cast(-8.772, torch.float32, 0)


def test_cast_as_tensor_numpy_bool():
	_test_cast(numpy.array(False), torch.bool, 0)
	_test_cast(numpy.array(True), torch.bool, 0)


def test_cast_as_tensor_numpy_int():
	_test_cast(numpy.array(5), torch.int64, 0)
	_test_cast(numpy.array(0), torch.int64, 0)
	_test_cast(numpy.array(-1), torch.int64, 0)


def test_cast_as_tensor_numpy_float():
	_test_cast(numpy.array(1.2), torch.float64, 0)
	_test_cast(numpy.array(0.0), torch.float64, 0)
	_test_cast(numpy.array(-8.772), torch.float64, 0)


def test_cast_as_tensor_torch_bool():
	_test_cast(torch.tensor(False), torch.bool, 0)
	_test_cast(torch.tensor(True), torch.bool, 0)


def test_cast_as_tensor_torch_int():
	_test_cast(torch.tensor(5), torch.int64, 0)
	_test_cast(torch.tensor(0), torch.int64, 0)
	_test_cast(torch.tensor(-1), torch.int64, 0)


def test_cast_as_tensor_torch_float():
	_test_cast(torch.tensor(1.2), torch.float32, 0)
	_test_cast(torch.tensor(0.0), torch.float32, 0)
	_test_cast(torch.tensor(-8.772), torch.float32, 0)


def test_cast_as_tensor_check_wrong():
	assert_raises(AssertionError, _test_cast, True, torch.int64, 0)
	assert_raises(AssertionError, _test_cast, True, torch.bool, 1)
	assert_raises(AssertionError, _test_cast, 1, torch.int32, 0)
	assert_raises(AssertionError, _test_cast, 1, torch.int64, 1)
	assert_raises(AssertionError, _test_cast, 1.2, torch.int64, 0)
	assert_raises(AssertionError, _test_cast, 1.2, torch.float64, 1)



def test_cast_as_tensor_numpy_bool_1d():
	_test_cast(numpy.array([True, False, True, True]), torch.bool, 1)
	_test_cast(numpy.array([True, True, True]), torch.bool, 1)
	_test_cast(numpy.array([False]), torch.bool, 1)


def test_cast_as_tensor_numpy_int_1d():
	_numpy_dtypes = numpy.int32, numpy.int64
	_torch_dtypes = torch.int32, torch.int64

	for _dtype1, _dtype2 in zip(_numpy_dtypes, _torch_dtypes):
		_test_cast(numpy.array([1, 2, 3], dtype=_dtype1), _dtype2, 1)
		_test_cast(numpy.array([0, -3, 0], dtype=_dtype1), _dtype2, 1)
		_test_cast(numpy.array([0], dtype=_dtype1), _dtype2, 1)


def test_cast_as_tensor_numpy_float_1d():
	_numpy_dtypes = numpy.float16, numpy.float32, numpy.float64
	_torch_dtypes = torch.float16, torch.float32, torch.float64

	for _dtype1, _dtype2 in zip(_numpy_dtypes, _torch_dtypes):
		_test_cast(numpy.array([1.2, 2.0, 3.1], dtype=_dtype1), _dtype2, 1)
		_test_cast(numpy.array([0.0, -3.0, 0.0], dtype=_dtype1), _dtype2, 1)
		_test_cast(numpy.array([0.0], dtype=_dtype1), _dtype2, 1)


def test_cast_as_tensor_numpy_check_wrong_1d():
	x1 = numpy.array([True, True, False])
	x2 = numpy.array([1, 2, 3])
	x3 = numpy.array([1.0, 1.1, 1.1])

	assert_raises(AssertionError, _test_cast, x1, torch.int64, 1)
	assert_raises(AssertionError, _test_cast, x1, torch.bool, 0)
	assert_raises(AssertionError, _test_cast, x2, torch.int32, 1)
	assert_raises(AssertionError, _test_cast, x2, torch.int64, 0)
	assert_raises(AssertionError, _test_cast, x3, torch.int64, 1)
	assert_raises(AssertionError, _test_cast, x3, torch.float64, 0)



def test_cast_as_tensor_numpy_bool_2d():
	_test_cast(numpy.array([[True, False, True], [True, True, False]]), 
		torch.bool, 2)
	_test_cast(numpy.array([[True, True, True]]), torch.bool, 2)
	_test_cast(numpy.array([[False]]), torch.bool, 2)


def test_cast_as_tensor_numpy_int_2d():
	_numpy_dtypes = numpy.int32, numpy.int64
	_torch_dtypes = torch.int32, torch.int64

	for _dtype1, _dtype2 in zip(_numpy_dtypes, _torch_dtypes):
		_test_cast(numpy.array([[1, 2, 3], [3, 2, 1]], dtype=_dtype1), 
			_dtype2, 2)
		_test_cast(numpy.array([[0, -3, 0]], dtype=_dtype1), _dtype2, 2)
		_test_cast(numpy.array([[0]], dtype=_dtype1), _dtype2, 2)


def test_cast_as_tensor_numpy_float_2d():
	_numpy_dtypes = numpy.float16, numpy.float32, numpy.float64
	_torch_dtypes = torch.float16, torch.float32, torch.float64

	for _dtype1, _dtype2 in zip(_numpy_dtypes, _torch_dtypes):
		_test_cast(numpy.array([[1.2, 2.0, 3.1], [-1.4, -1.3, -0.0]], 
			dtype=_dtype1), _dtype2, 2)
		_test_cast(numpy.array([[0.0, -3.0, 0.0]], dtype=_dtype1), _dtype2, 2)
		_test_cast(numpy.array([[0.0]], dtype=_dtype1), _dtype2, 2)


def test_cast_as_tensor_numpy_check_wrong_2d():
	x1 = numpy.array([[True, True, False]])
	x2 = numpy.array([[1, 2, 3]])
	x3 = numpy.array([[1.0, 1.1, 1.1]])

	assert_raises(AssertionError, _test_cast, x1, torch.int64, 2)
	assert_raises(AssertionError, _test_cast, x1, torch.bool, 1)
	assert_raises(AssertionError, _test_cast, x2, torch.int32, 2)
	assert_raises(AssertionError, _test_cast, x2, torch.int64, 1)
	assert_raises(AssertionError, _test_cast, x3, torch.int64, 2)
	assert_raises(AssertionError, _test_cast, x3, torch.float64, 1)



def test_cast_as_tensor_torch_bool_1d():
	_test_cast(torch.tensor([True, False, True, True]), torch.bool, 1)
	_test_cast(torch.tensor([True, True, True]), torch.bool, 1)
	_test_cast(torch.tensor([False]), torch.bool, 1)


def test_cast_as_tensor_torch_int_1d():
	_torch_dtypes = torch.int32, torch.int64

	for _dtype in _torch_dtypes:
		_test_cast(torch.tensor([1, 2, 3], dtype=_dtype), _dtype, 1)
		_test_cast(torch.tensor([0, -3, 0], dtype=_dtype), _dtype, 1)
		_test_cast(torch.tensor([0], dtype=_dtype), _dtype, 1)


def test_cast_as_tensor_torch_float_1d():
	_torch_dtypes = torch.float16, torch.float32, torch.float64

	for _dtype in _torch_dtypes:
		_test_cast(torch.tensor([1.2, 2.0, 3.1], dtype=_dtype), _dtype, 1)
		_test_cast(torch.tensor([0.0, -3.0, 0.0], dtype=_dtype), _dtype, 1)
		_test_cast(torch.tensor([0.0], dtype=_dtype), _dtype, 1)


def test_cast_as_tensor_torch_check_wrong_1d():
	x1 = torch.tensor([True, True, False])
	x2 = torch.tensor([1, 2, 3])
	x3 = torch.tensor([1.0, 1.1, 1.1])

	assert_raises(AssertionError, _test_cast, x1, torch.int64, 1)
	assert_raises(AssertionError, _test_cast, x1, torch.bool, 0)
	assert_raises(AssertionError, _test_cast, x2, torch.int32, 1)
	assert_raises(AssertionError, _test_cast, x2, torch.int64, 0)
	assert_raises(AssertionError, _test_cast, x3, torch.int64, 1)
	assert_raises(AssertionError, _test_cast, x3, torch.float64, 0)



def test_cast_as_tensor_torch_bool_2d():
	_test_cast(torch.tensor([[True, False, True], [True, True, False]]), 
		torch.bool, 2)
	_test_cast(torch.tensor([[True, True, True]]), torch.bool, 2)
	_test_cast(torch.tensor([[False]]), torch.bool, 2)


def test_cast_as_tensor_torch_int_2d():
	_torch_dtypes = torch.int32, torch.int64

	for _dtype in _torch_dtypes:
		_test_cast(torch.tensor([[1, 2, 3], [3, 2, 1]], dtype=_dtype), 
			_dtype, 2)
		_test_cast(torch.tensor([[0, -3, 0]], dtype=_dtype), _dtype, 2)
		_test_cast(torch.tensor([[0]], dtype=_dtype), _dtype, 2)


def test_cast_as_tensor_torch_float_2d():
	_torch_dtypes = torch.float16, torch.float32, torch.float64

	for _dtype in _torch_dtypes:
		_test_cast(torch.tensor([[1.2, 2.0, 3.1], [-1.4, -1.3, -0.0]], 
			dtype=_dtype), _dtype, 2)
		_test_cast(torch.tensor([[0.0, -3.0, 0.0]], dtype=_dtype), _dtype, 2)
		_test_cast(torch.tensor([[0.0]], dtype=_dtype), _dtype, 2)


def test_cast_as_tensor_torch_check_wrong_2d():
	x1 = torch.tensor([[True, True, False]])
	x2 = torch.tensor([[1, 2, 3]])
	x3 = torch.tensor([[1.0, 1.1, 1.1]])

	assert_raises(AssertionError, _test_cast, x1, torch.int64, 2)
	assert_raises(AssertionError, _test_cast, x1, torch.bool, 1)
	assert_raises(AssertionError, _test_cast, x2, torch.int32, 2)
	assert_raises(AssertionError, _test_cast, x2, torch.int64, 1)
	assert_raises(AssertionError, _test_cast, x3, torch.int64, 2)
	assert_raises(AssertionError, _test_cast, x3, torch.float64, 1)



#

def _test_update(inertia):
	x1 = torch.tensor([1.0, 1.4, 1.8, -1.1, 0.0])
	x2 = torch.tensor([2.2, 8.2, 0.1, 105.2, 0.0])
	y = x1 * inertia + x2 * (1-inertia) 

	_update_parameter(x1, x2, inertia=inertia)
	assert_array_almost_equal(x1, y)


def test_update_parameter():
	_test_update(inertia=0.0)


def test_update_parameter_inertia():
	_test_update(inertia=0.1)
	_test_update(inertia=0.5)
	_test_update(inertia=1.0)



#



def test_check_parameters_min_values_bool():
	x = torch.tensor([True, True, False], dtype=torch.bool)
	dtypes = [torch.bool]

	_check_parameter(x, "x", min_value=0)
	_check_parameter(x, "x", min_value=-1.0)

	assert_raises(ValueError, _check_parameter, x, "x", min_value=1)
	assert_raises(ValueError, _check_parameter, x, "x", min_value=1000.0)


def test_check_parameters_min_values_int():
	x = torch.tensor([1, 6, 24], dtype=torch.int32)
	dtypes = [torch.bool]

	_check_parameter(x, "x", min_value=1)
	_check_parameter(x, "x", min_value=-1.0)

	assert_raises(ValueError, _check_parameter, x, "x", min_value=2)
	assert_raises(ValueError, _check_parameter, x, "x", min_value=25.0)


def test_check_parameters_min_values_float():
	x = torch.tensor([1, 6, 24], dtype=torch.float32)
	dtypes = [torch.bool]

	_check_parameter(x, "x", min_value=1)
	_check_parameter(x, "x", min_value=-1.0)

	assert_raises(ValueError, _check_parameter, x, "x", min_value=2)
	assert_raises(ValueError, _check_parameter, x, "x", min_value=25.0)


def test_check_parameters_max_values_bool():
	x = torch.tensor([True, True, False], dtype=torch.bool)
	dtypes = [torch.bool]

	_check_parameter(x, "x", max_value=1)
	_check_parameter(x, "x", max_value=100.0)

	assert_raises(ValueError, _check_parameter, x, "x", max_value=0)
	assert_raises(ValueError, _check_parameter, x, "x", max_value=-2.7)


def test_check_parameters_max_values_int():
	x = torch.tensor([1, 6, 24], dtype=torch.int32)
	dtypes = [torch.bool]

	_check_parameter(x, "x", max_value=24)
	_check_parameter(x, "x", max_value=24.1)

	assert_raises(ValueError, _check_parameter, x, "x", max_value=2)
	assert_raises(ValueError, _check_parameter, x, "x", max_value=8.0)


def test_check_parameters_max_values_float():
	x = torch.tensor([1, 6, 24], dtype=torch.float32)
	dtypes = [torch.bool]

	_check_parameter(x, "x", max_value=24)
	_check_parameter(x, "x", max_value=24.1)

	assert_raises(ValueError, _check_parameter, x, "x", max_value=2)
	assert_raises(ValueError, _check_parameter, x, "x", max_value=8.0)


def test_check_parameters_minmax_values_float():
	x = torch.tensor([1.1, 2.3, 7.8], dtype=torch.float32)
	dtypes = [torch.bool]

	_check_parameter(x, "x", min_value=1.0, max_value=24)

	assert_raises(ValueError, _check_parameter, x, "x", min_value=1.2, 
		max_value=24)
	assert_raises(ValueError, _check_parameter, x, "x", min_value=0.0,
		max_value=6)


def test_check_parameters_value_set_bool():
	x = torch.tensor([True, True, True], dtype=torch.bool)
	value_set = [True]

	_check_parameter(x, "x", value_set=tuple(value_set))
	_check_parameter(x, "x", value_set=list(value_set))

	assert_raises(ValueError, _check_parameter, x, "x", value_set=[False])
	assert_raises(ValueError, _check_parameter, x, "x", value_set=[5.2])


def test_check_parameters_value_set_int():
	x = torch.tensor([2, 6, 24], dtype=torch.int32)
	value_set = [2, 6, 24, 26]

	_check_parameter(x, "x", value_set=tuple(value_set))
	_check_parameter(x, "x", value_set=list(value_set))

	assert_raises(ValueError, _check_parameter, x, "x", value_set=[True, False])
	assert_raises(ValueError, _check_parameter, x, "x", value_set=[5.2, 1, 6])


def test_check_parameters_value_set_float():
	x = torch.tensor([1.1, 6.0, 24.3], dtype=torch.float32)
	value_set = [1.1, 6.0, 24.3, 17.8]

	_check_parameter(x, "x", value_set=tuple(value_set))
	_check_parameter(x, "x", value_set=list(value_set))

	assert_raises(ValueError, _check_parameter, x, "x", value_set=[True, False])
	assert_raises(ValueError, _check_parameter, x, "x", value_set=[5.2, 1, 6])


def test_check_parameters_dtypes_bool():
	x = torch.tensor([True, True, False], dtype=torch.bool)
	dtypes = [torch.bool]

	_check_parameter(x, "x", dtypes=tuple(dtypes))
	_check_parameter(x, "x", dtypes=list(dtypes))
	_check_parameter(x, "x", dtypes=set(dtypes))

	assert_raises(ValueError, _check_parameter, x, "x", dtypes=[torch.int64])
	assert_raises(ValueError, _check_parameter, x, "x", dtypes=[torch.float64])


def test_check_parameters_dtypes_int():
	x = torch.tensor([1, 2, 3], dtype=torch.int32)
	dtypes = [torch.int32, torch.int64]

	_check_parameter(x, "x", dtypes=tuple(dtypes))
	_check_parameter(x, "x", dtypes=list(dtypes))
	_check_parameter(x, "x", dtypes=set(dtypes))

	assert_raises(ValueError, _check_parameter, x, "x", dtypes=[torch.int64])
	assert_raises(ValueError, _check_parameter, x, "x", dtypes=[torch.float32])


def test_check_parameters_dtypes_float():
	x = torch.tensor([1, 2, 3], dtype=torch.float32)
	dtypes = [torch.float32, torch.float64]

	_check_parameter(x, "x", dtypes=tuple(dtypes))
	_check_parameter(x, "x", dtypes=list(dtypes))
	_check_parameter(x, "x", dtypes=set(dtypes))

	assert_raises(ValueError, _check_parameter, x, "x", dtypes=[torch.int64])
	assert_raises(ValueError, _check_parameter, x, "x", dtypes=[torch.float64])


def test_check_parameters_ndim_0():
	x = torch.tensor(1.1)

	_check_parameter(x, "x", ndim=0)
	assert_raises(ValueError, _check_parameter, x, "x", ndim=1)
	assert_raises(ValueError, _check_parameter, x, "x", ndim=2)


def test_check_parameters_ndim_1():
	x = torch.tensor([1.1])

	_check_parameter(x, "x", ndim=1)
	assert_raises(ValueError, _check_parameter, x, "x", ndim=0)
	assert_raises(ValueError, _check_parameter, x, "x", ndim=2)


def test_check_parameters_ndim_2():
	x = torch.tensor([[1.1]])

	_check_parameter(x, "x", ndim=2)
	assert_raises(ValueError, _check_parameter, x, "x", ndim=1)
	assert_raises(ValueError, _check_parameter, x, "x", ndim=0)


def test_check_parameters_shape():
	x = torch.tensor([[1.1]])

	_check_parameter(x, "x", shape=(1, 1))
	_check_parameter(x, "x", shape=(-1, 1))
	_check_parameter(x, "x", shape=(1, -1))
	_check_parameter(x, "x", shape=(-1, -1))
	assert_raises(ValueError, _check_parameter, x, "x", shape=(2, 1))
	assert_raises(ValueError, _check_parameter, x, "x", shape=(1, 2))
	assert_raises(ValueError, _check_parameter, x, "x", shape=(1,))
	assert_raises(ValueError, _check_parameter, x, "x", shape=(1, 2, 1))
	assert_raises(ValueError, _check_parameter, x, "x", shape=(1, 2, -1))

	x = torch.tensor([
		[1.1, 1.2, 1.3, 1.8],
		[2.1, 1.1, 1.4, 0.9]
	])

	_check_parameter(x, "x", shape=(2, 4))
	_check_parameter(x, "x", shape=(-1, 4))
	_check_parameter(x, "x", shape=(2, -1))
	_check_parameter(x, "x", shape=(-1, -1))
	assert_raises(ValueError, _check_parameter, x, "x", shape=(2, 1))
	assert_raises(ValueError, _check_parameter, x, "x", shape=(1, 2))
	assert_raises(ValueError, _check_parameter, x, "x", shape=(1,))
	assert_raises(ValueError, _check_parameter, x, "x", shape=(1, 2, 1))
	assert_raises(ValueError, _check_parameter, x, "x", shape=(1, 2, -1))
	assert_raises(ValueError, _check_parameter, x, "x", shape=(2, -1, -1))
	