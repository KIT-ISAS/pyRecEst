"""Regression tests for PyTorch backend take-axis validation."""

from __future__ import annotations

import importlib.util

import pytest

from tests.support.backend_runner import run_backend_code


@pytest.mark.backend_portable
def test_pytorch_take_rejects_non_integer_axes_like_numpy():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        "pytorch",
        """
import numpy as np
import torch

import pyrecest.backend as backend
import pyrecest._backend.pytorch as raw_pytorch


def assert_axis_type_error(take_func, values, axis):
    try:
        take_func(values, [0], axis=axis)
    except TypeError:
        return
    raise AssertionError(f"take accepted non-integer axis {axis!r}")


values = backend.asarray([[0, 1], [2, 3]])
raw_values = raw_pytorch.array([[0, 1], [2, 3]])

for invalid_axis in (True, 1.0, "1", np.array(True), torch.tensor(True)):
    assert_axis_type_error(backend.take, values, invalid_axis)
    assert_axis_type_error(raw_pytorch.take, raw_values, invalid_axis)

assert backend.to_numpy(backend.take(values, [0], axis=np.array(0))).tolist() == [[0, 1]]
assert raw_pytorch.to_numpy(raw_pytorch.take(raw_values, [0], axis=np.array(1))).tolist() == [
    [0],
    [2],
]
""",
    )

    assert result.returncode == 0, result.stderr
