"""Regression tests for PyTorch backend linear solve coercion."""

from __future__ import annotations

import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code

_CHECK = """
import pyrecest.backend as backend

solution = backend.linalg.solve([[2, 0], [0, 4]], [2, 8])
assert tuple(backend.shape(solution)) == (2,)
assert solution.dtype == backend.float64
assert backend.to_numpy(solution).tolist() == [1.0, 2.0]

matrix_rhs = backend.linalg.solve(
    backend.array([[2.0, 0.0], [0.0, 4.0]], dtype=backend.float32),
    backend.array([[2.0, 4.0], [8.0, 12.0]], dtype=backend.float64),
)
assert matrix_rhs.dtype == backend.float64
assert backend.to_numpy(matrix_rhs).tolist() == [[1.0, 2.0], [2.0, 3.0]]
print("ok")
"""


@pytest.mark.backend_portable
def test_pytorch_linalg_solve_accepts_array_like_inputs_and_promotes_dtype():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code("pytorch", _CHECK)

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
