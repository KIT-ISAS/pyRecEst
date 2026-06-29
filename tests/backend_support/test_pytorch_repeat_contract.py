"""Regression tests for PyTorch backend repeat semantics."""

from __future__ import annotations

import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code


@pytest.mark.backend_portable
def test_pytorch_repeat_accepts_numpy_axis_keyword_and_array_counts():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        "pytorch",
        """
from pyrecest.backend import asarray, repeat, to_numpy
import pyrecest._backend.pytorch as pytorch_backend

values = asarray([[1, 2], [3, 4]])

axis_zero = repeat(values, 2, axis=0)
assert to_numpy(axis_zero).tolist() == [[1, 2], [1, 2], [3, 4], [3, 4]]

axis_last = repeat(values, [1, 2], axis=-1)
assert to_numpy(axis_last).tolist() == [[1, 2, 2], [3, 4, 4]]

flattened = repeat(values, [1, 2, 1, 0])
assert to_numpy(flattened).tolist() == [1, 2, 2, 3]

raw_backend_result = pytorch_backend.repeat([[1, 2]], 2, axis=1)
assert to_numpy(raw_backend_result).tolist() == [[1, 1, 2, 2]]
""",
    )

    assert result.returncode == 0, result.stderr
