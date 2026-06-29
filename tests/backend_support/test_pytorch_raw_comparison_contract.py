"""Regression tests for raw PyTorch backend comparison helpers."""

from __future__ import annotations

import importlib
import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code


@pytest.mark.backend_portable
def test_raw_pytorch_comparisons_accept_numpy_style_array_like_inputs():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        "pytorch",
        """
import importlib
from pyrecest.backend import asarray, to_numpy

raw_pytorch = importlib.import_module("pyrecest._backend.pytorch")

greater_result = raw_pytorch.greater([1, 2, 3], asarray([0, 2, 4]))
less_result = raw_pytorch.less(asarray([1, 2, 3]), [0, 2, 4])
logical_result = raw_pytorch.logical_or([True, False], asarray([False, True]))

assert to_numpy(greater_result).tolist() == [True, False, False]
assert to_numpy(less_result).tolist() == [False, False, True]
assert to_numpy(logical_result).tolist() == [True, True]
""",
    )

    assert result.returncode == 0, result.stderr
