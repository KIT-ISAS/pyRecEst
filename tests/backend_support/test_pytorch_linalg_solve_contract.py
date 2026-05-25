"""Regression tests for PyTorch backend linear-solve input coercion."""

from __future__ import annotations

import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code


@pytest.mark.backend_portable
def test_pytorch_linalg_solve_accepts_array_like_and_mixed_dtypes():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        "pytorch",
        """
import pyrecest.backend as backend

system = [[2, 0], [0, 4]]
rhs = backend.asarray([2.0, 8.0], dtype=backend.float32)
solution = backend.linalg.solve(system, rhs)
expected = backend.asarray([1.0, 2.0], dtype=solution.dtype)
assert tuple(solution.shape) == (2,)
assert backend.allclose(solution, expected)
assert solution.dtype == backend.float64
print("ok")
""",
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
