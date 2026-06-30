import numpy as np

from pyrecest._backend import _common


def test_normalize_reduction_axes_accepts_numpy_scalar_integer_axis():
    assert _common._normalize_reduction_axes(np.asarray(0), 2) == (0,)
