"""Regression tests for the JAX backend ``to_ndarray`` helper."""

from __future__ import annotations

import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code


@pytest.mark.backend_portable
def test_jax_to_ndarray_adds_all_missing_dimensions_and_rejects_too_many():
    if importlib.util.find_spec("jax") is None:
        pytest.skip("JAX is not installed")

    result = run_backend_code(
        "jax",
        """
import pyrecest.backend as backend

expanded = backend.to_ndarray(backend.asarray([1.0, 2.0]), 3)
assert expanded.shape == (1, 1, 2)

try:
    backend.to_ndarray(backend.zeros((1, 2, 3)), 2)
except ValueError as exc:
    assert "ndim" in str(exc)
else:
    raise AssertionError("expected ValueError for reducing ndim")
""",
    )

    assert result.returncode == 0, result.stderr
