"""Regression tests for PyTorch arange dtype normalization."""

from __future__ import annotations

import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code


@pytest.mark.backend_portable
def test_public_pytorch_arange_accepts_numpy_dtype_aliases():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        "pytorch",
        """
import numpy as np
import pyrecest.backend as backend

from_numpy_type = backend.arange(3, dtype=np.float32)
from_numpy_dtype = backend.arange(3, dtype=np.dtype("float64"))
from_torch_string = backend.arange(3, dtype="torch.float64")

assert from_numpy_type.dtype == backend.float32
assert from_numpy_dtype.dtype == backend.float64
assert from_torch_string.dtype == backend.float64
assert backend.to_numpy(from_numpy_type).tolist() == [0.0, 1.0, 2.0]
assert backend.to_numpy(from_numpy_dtype).tolist() == [0.0, 1.0, 2.0]
assert backend.to_numpy(from_torch_string).tolist() == [0.0, 1.0, 2.0]
""",
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.backend_portable
def test_raw_pytorch_arange_is_patched_under_non_pytorch_backend():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        "numpy",
        """
import numpy as np
import pyrecest  # noqa: F401
import pyrecest._backend.pytorch as raw_pytorch

values = raw_pytorch.arange(3, dtype=np.float64)

assert values.dtype == raw_pytorch.float64
assert values.tolist() == [0.0, 1.0, 2.0]
""",
    )

    assert result.returncode == 0, result.stderr
