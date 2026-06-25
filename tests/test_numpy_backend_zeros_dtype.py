import numpy as np

from pyrecest._backend import numpy as numpy_backend


def test_numpy_zeros_preserves_explicit_positional_dtype():
    result = numpy_backend.zeros((2,), np.float32)

    assert result.dtype == np.dtype(np.float32)


def test_numpy_zeros_uses_default_dtype_when_dtype_is_omitted():
    result = numpy_backend.zeros((2,))

    assert result.dtype == numpy_backend.get_default_dtype()
