import importlib.util

import pytest

from tests.support.backend_runner import run_backend_code


@pytest.mark.backend_portable
def test_jax_fftshift_accepts_numpy_integer_scalar_axes():
    if importlib.util.find_spec("jax") is None:
        pytest.skip("jax is not installed")

    code = """
import numpy as np

import pyrecest.backend as backend

matrix = [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]

shifted_axis = backend.fft.fftshift(matrix, axes=np.int64(1))
assert backend.to_numpy(shifted_axis).tolist() == [[2.0, 0.0, 1.0], [5.0, 3.0, 4.0]]

unshifted_axis = backend.fft.ifftshift(matrix, axes=np.int64(1))
assert backend.to_numpy(unshifted_axis).tolist() == [[1.0, 2.0, 0.0], [4.0, 5.0, 3.0]]
"""
    result = run_backend_code("jax", code)
    assert result.returncode == 0, result.stderr
