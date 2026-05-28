"""Regression test for JAX random.normal shape argument handling."""

from __future__ import annotations

import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code

_CHECK = """
import numpy as np
import pyrecest.backend as backend
from pyrecest.backend import random

samples_from_numpy_scalar = random.normal(np.int64(3))
samples_from_numpy_tuple = random.normal((np.int64(2), 3))
samples_from_python_tuple = random.normal((2, 3))

assert tuple(backend.shape(samples_from_numpy_scalar)) == (3,)
assert tuple(backend.shape(samples_from_numpy_tuple)) == (2, 3)
assert tuple(backend.shape(samples_from_python_tuple)) == (2, 3)
print("ok")
"""


@pytest.mark.backend_portable
def test_jax_normal_accepts_numpy_integer_shape_arguments():
    if importlib.util.find_spec("jax") is None:
        pytest.skip("JAX is not installed")

    result = run_backend_code("jax", _CHECK)

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
