"""Regression tests for PyTorch linalg.solve device preservation."""

from __future__ import annotations

import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code


@pytest.mark.backend_portable
def test_pytorch_linalg_solve_prefers_non_cpu_tensor_device():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        "pytorch",
        """
import torch
import pyrecest.backend as backend

matrix = torch.eye(2)
rhs = torch.ones(2, device="meta")
solution = backend.linalg.solve(matrix, rhs)

assert solution.device.type == "meta", solution.device
assert tuple(solution.shape) == (2,)
""",
    )

    assert result.returncode == 0, result.stderr
