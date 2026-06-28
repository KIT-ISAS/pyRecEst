"""Regression tests for PyTorch backend comparison helpers."""

from __future__ import annotations

import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code


@pytest.mark.backend_portable
def test_pytorch_comparisons_accept_numpy_style_array_like_inputs():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        "pytorch",
        """
from pyrecest.backend import asarray, greater, less, to_numpy

greater_result = greater([1, 2, 3], [0, 2, 4])
less_result = less([1, 2, 3], [0, 2, 4])
mixed_result = greater(asarray([1.0, 2.0, 3.0]), [0.5, 2.0, 5.0])

assert to_numpy(greater_result).tolist() == [True, False, False]
assert to_numpy(less_result).tolist() == [False, False, True]
assert to_numpy(mixed_result).tolist() == [True, False, False]
""",
    )

    assert result.returncode == 0, result.stderr
