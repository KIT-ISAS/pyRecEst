import numpy as np

def test_array_none():
    value = np.array([[None]], dtype=object)
    assert value.shape == (1, 1)
