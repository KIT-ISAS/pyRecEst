"""Regression tests for raw PyTorch backend comparison helpers."""

from __future__ import annotations

import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code


_RAW_COMPARISON_CODE = """
import pyrecest._backend.pytorch as raw_pytorch

greater_result = raw_pytorch.greater([1, 2, 3], raw_pytorch.asarray([0, 2, 4]))
less_result = raw_pytorch.less(raw_pytorch.asarray([1, 2, 3]), [0, 2, 4])
logical_result = raw_pytorch.logical_or(
    [True, False],
    raw_pytorch.asarray([False, True]),
)

assert raw_pytorch.to_numpy(greater_result).tolist() == [True, False, False]
assert raw_pytorch.to_numpy(less_result).tolist() == [False, False, True]
assert raw_pytorch.to_numpy(logical_result).tolist() == [True, True]
"""


@pytest.mark.backend_portable
@pytest.mark.parametrize("backend_name", ["pytorch", "numpy"])
def test_raw_pytorch_comparisons_accept_numpy_style_array_like_inputs(backend_name):
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(backend_name, _RAW_COMPARISON_CODE)

    assert result.returncode == 0, result.stderr
