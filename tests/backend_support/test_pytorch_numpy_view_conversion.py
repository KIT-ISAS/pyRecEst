"""Regression tests for PyTorch backend NumPy view conversion."""

from __future__ import annotations

import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code


@pytest.mark.backend_portable
def test_pytorch_backend_accepts_negative_stride_numpy_views():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        "pytorch",
        """
import numpy as np
import pyrecest.backend as backend
from pyrecest._backend.pytorch._common import from_numpy

vector_view = np.arange(5.0)[::-1]
vector_tensor = from_numpy(vector_view)
assert backend.to_numpy(vector_tensor).tolist() == [4.0, 3.0, 2.0, 1.0, 0.0]

matrix_view = np.arange(6.0).reshape(2, 3)[:, ::-1]
matrix_tensor = backend.array(matrix_view)
assert backend.to_numpy(matrix_tensor).tolist() == [[2.0, 1.0, 0.0], [5.0, 4.0, 3.0]]
""",
    )

    assert result.returncode == 0, result.stderr
