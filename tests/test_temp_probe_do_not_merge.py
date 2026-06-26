import numpy as np
import pyrecest.backend as backend


def _mapped_output(value):
    if value == 0:
        return value
    return value + 0.5


def test_numpy_vmap_preserves_later_float_result():
    if backend.__backend_name__ != "numpy":
        return

    result = backend.vmap(_mapped_output)(np.array([0, 1]))

    assert backend.to_numpy(result).tolist() == [0.0, 1.5]
