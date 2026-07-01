import numpy as np
import pytest

pytest.importorskip("jax")
from pyrecest._backend import jax as jax_backend  # noqa: E402


def test_linalg_norm_accepts_numpy_array_axes():
    values = jax_backend.array([[3.0, 4.0], [0.0, 5.0]])
    expected = jax_backend.array([5.0, 5.0])

    for axis in (np.array(1), np.array([1])):
        result = jax_backend.linalg.norm(values, axis=axis)

        assert jax_backend.allclose(result, expected)
