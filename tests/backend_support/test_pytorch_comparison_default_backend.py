"""Regression coverage for PyTorch comparison helpers with another public backend."""

from __future__ import annotations

import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code


@pytest.mark.backend_portable
def test_pytorch_comparison_helpers_work_with_numpy_public_backend():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        "numpy",
        """
import importlib

pt = importlib.import_module("pyrecest._backend.pytorch")

greater_result = pt.greater([1, 2, 3], pt.array([0, 2, 4]))
less_result = pt.less(pt.array([1, 2, 3]), [0, 2, 4])
logical_result = pt.logical_or([True, False], pt.array([False, True]))

assert pt.to_numpy(greater_result).tolist() == [True, False, False]
assert pt.to_numpy(less_result).tolist() == [False, False, True]
assert pt.to_numpy(logical_result).tolist() == [True, True]
""",
    )

    assert result.returncode == 0, result.stderr
