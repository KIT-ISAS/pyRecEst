"""Regression tests for raw JAX matmul out handling under another backend."""

from __future__ import annotations

import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code


@pytest.mark.backend_portable
def test_raw_jax_matmul_honors_out_with_numpy_public_backend():
    if importlib.util.find_spec("jax") is None:
        pytest.skip("JAX is not installed")

    result = run_backend_code(
        "numpy",
        """
import importlib

raw_jax = importlib.import_module("pyrecest._backend.jax")

left = [[1.0, 2.0], [3.0, 4.0]]
right = [[1.0, 0.0], [0.0, 1.0]]

out = raw_jax.zeros((2, 2))
returned = raw_jax.matmul(left, right, out=out)
assert raw_jax.to_numpy(returned).tolist() == [[1.0, 2.0], [3.0, 4.0]]

bad_out = raw_jax.zeros((1, 1))
try:
    raw_jax.matmul(left, right, out=bad_out)
except (TypeError, ValueError):
    pass
else:
    raise AssertionError("raw JAX matmul ignored incompatible out shape")
""",
    )

    assert result.returncode == 0, result.stderr
