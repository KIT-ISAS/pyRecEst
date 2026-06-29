"""Regression coverage for PyTorch degree/radian helpers."""

from __future__ import annotations

import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code


@pytest.mark.backend_portable
@pytest.mark.parametrize("public_backend", ["pytorch", "numpy"])
def test_pytorch_degree_radian_helpers_accept_array_like_inputs(public_backend):
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        public_backend,
        f"""
import importlib
import torch

pt = importlib.import_module("pyrecest._backend.pytorch")

degrees = [0, 90, 180]
radians = pt.deg2rad(degrees)
expected_radians = torch.tensor(
    [0.0, torch.pi / 2, torch.pi], dtype=radians.dtype, device=radians.device
)
assert torch.allclose(radians, expected_radians)

out = torch.empty_like(radians)
returned = pt.rad2deg(radians, out=out)
assert returned is out
assert torch.allclose(
    out, torch.tensor(degrees, dtype=out.dtype, device=out.device)
)

if {public_backend!r} == "pytorch":
    import pyrecest.backend as backend

    public_radians = backend.deg2rad(degrees)
    assert torch.allclose(public_radians, expected_radians)
""",
    )

    assert result.returncode == 0, result.stderr
