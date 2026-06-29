"""Regression coverage for raw PyTorch assignment helpers with another public backend."""

from __future__ import annotations

import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code


@pytest.mark.backend_portable
def test_raw_pytorch_assignment_accepts_scalar_tensor_index_with_numpy_public_backend():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        "numpy",
        """
import importlib

public_backend = importlib.import_module("pyrecest.backend")
pt = importlib.import_module("pyrecest._backend.pytorch")
torch = importlib.import_module("torch")

assert public_backend.__backend_name__ == "numpy"

assigned = pt.assignment(pt.array([1, 2, 3]), 9, torch.tensor(1))
summed = pt.assignment_by_sum(pt.array([1, 2, 3]), 7, torch.tensor(1))

assert assigned.tolist() == [1, 9, 3]
assert summed.tolist() == [1, 9, 3]
""",
    )

    assert result.returncode == 0, result.stderr
