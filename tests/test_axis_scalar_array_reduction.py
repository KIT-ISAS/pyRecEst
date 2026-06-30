import numpy as np


def test_placeholder_axis_scalar_array():
    assert np.array(0).shape == ()
