"""Regression tests for PyTorch array_equal NumPy keyword semantics."""

from __future__ import annotations

import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code


def _skip_without_torch() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")


@pytest.mark.backend_portable
def test_raw_pytorch_array_equal_accepts_equal_nan_under_numpy_backend():
    _skip_without_torch()

    result = run_backend_code(
        "numpy",
        """
import importlib

raw_pytorch = importlib.import_module("pyrecest._backend.pytorch")

assert raw_pytorch.array_equal([1.0, 2.0], [1.0, 2.0])
assert not raw_pytorch.array_equal([float("nan")], [float("nan")])
assert raw_pytorch.array_equal(
    [float("nan"), 1.0],
    [float("nan"), 1.0],
    equal_nan=True,
)
assert not raw_pytorch.array_equal(
    [float("nan"), 1.0],
    [float("nan"), 2.0],
    equal_nan=True,
)
assert raw_pytorch.array_equal(
    [complex(float("nan"), 1.0)],
    [complex(2.0, float("nan"))],
    equal_nan=True,
)
""",
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.backend_portable
def test_public_pytorch_array_equal_accepts_equal_nan_keyword():
    _skip_without_torch()

    result = run_backend_code(
        "pytorch",
        """
import pyrecest.backend as backend

assert backend.array_equal(
    [float("nan"), 1.0],
    [float("nan"), 1.0],
    equal_nan=True,
)
assert not backend.array_equal(
    [float("nan"), 1.0],
    [float("nan"), 2.0],
    equal_nan=True,
)
""",
    )

    assert result.returncode == 0, result.stderr
