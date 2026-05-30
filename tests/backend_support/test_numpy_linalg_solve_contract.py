"""Regression tests for shared NumPy-backend linear solve semantics."""

from __future__ import annotations

import pytest
from tests.support.backend_runner import run_backend_code


@pytest.mark.backend_portable
def test_numpy_linalg_solve_preserves_batched_matrix_rhs_shape_and_values():
    result = run_backend_code(
        "numpy",
        """
import pyrecest.backend as backend

A = backend.array([
    [[2.0, 0.0], [0.0, 4.0]],
    [[1.0, 0.0], [0.0, 5.0]],
])
B = backend.array([
    [[2.0, 4.0], [8.0, 12.0]],
    [[3.0, 6.0], [10.0, 15.0]],
])

X = backend.linalg.solve(A, B)

assert X.shape == (2, 2, 2)
assert backend.to_numpy(X).tolist() == [
    [[1.0, 2.0], [2.0, 3.0]],
    [[3.0, 6.0], [2.0, 3.0]],
]
""",
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.backend_portable
def test_numpy_linalg_solve_still_squeezes_batched_vector_rhs():
    result = run_backend_code(
        "numpy",
        """
import pyrecest.backend as backend

A = backend.array([
    [[2.0, 0.0], [0.0, 4.0]],
    [[1.0, 0.0], [0.0, 5.0]],
])
b = backend.array([[2.0, 8.0], [3.0, 10.0]])

x = backend.linalg.solve(A, b)

assert x.shape == (2, 2)
assert backend.to_numpy(x).tolist() == [[1.0, 2.0], [3.0, 2.0]]
""",
    )

    assert result.returncode == 0, result.stderr
