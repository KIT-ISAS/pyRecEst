import numpy as np

from pyrecest._backend import _common


def test_normalize_reduction_axes_accepts_scalar_array_axis():
    assert _common._normalize_reduction_axes(np.asarray(0), 2) == (0,)
