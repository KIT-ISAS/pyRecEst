"""Regression coverage for PyTorch covariance NumPy dtype semantics."""

from __future__ import annotations

import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code


@pytest.mark.backend_portable
def test_pytorch_cov_promotes_integer_observations_for_public_and_raw_backends():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    code = """
import pyrecest.backend as backend
import pyrecest._backend.pytorch as raw_pytorch

values = raw_pytorch.array([[1, 2, 3], [2, 4, 8]], dtype=raw_pytorch.int64)
expected = raw_pytorch.array(
    [[2.0 / 3.0, 2.0], [2.0, 56.0 / 9.0]],
    dtype=raw_pytorch.get_default_dtype(),
)

raw_result = raw_pytorch.cov(values, bias=True)
assert raw_result.dtype == raw_pytorch.get_default_dtype()
assert bool(raw_pytorch.allclose(raw_result, expected))

if backend.__backend_name__ == "pytorch":
    public_result = backend.cov([[1, 2, 3], [2, 4, 8]], bias=True)
    assert public_result.dtype == backend.get_default_dtype()
    assert bool(backend.allclose(public_result, expected))
"""

    for backend_name in ("numpy", "pytorch"):
        result = run_backend_code(backend_name, code)
        assert result.returncode == 0, result.stderr
