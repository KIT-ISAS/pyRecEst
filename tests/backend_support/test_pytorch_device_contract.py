"""Regression tests for PyTorch backend device preservation."""

from __future__ import annotations

import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code


@pytest.mark.backend_portable
def test_pytorch_imag_and_sparse_helpers_preserve_tensor_device():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        "pytorch",
        """
import torch
import pyrecest.backend as backend

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

real_tensor = torch.tensor([1.0, 2.0], dtype=torch.float64, device=device)
imag_part = backend.imag(real_tensor)
assert imag_part.device == real_tensor.device
assert imag_part.dtype == real_tensor.dtype
assert backend.to_numpy(imag_part).tolist() == [0.0, 0.0]

sparse_values = torch.tensor([5.0, 7.0], dtype=torch.float64, device=device)
sparse_array = backend.array_from_sparse(
    [(0, 1), (1, 0)],
    sparse_values,
    (2, 2),
)

assert sparse_array.device == sparse_values.device
assert sparse_array.dtype == sparse_values.dtype
assert backend.to_numpy(sparse_array).tolist() == [[0.0, 5.0], [7.0, 0.0]]
""",
    )

    assert result.returncode == 0, result.stderr


def test_pytorch_logical_helpers_prefer_non_cpu_tensor_device():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        "pytorch",
        """
import torch
import pyrecest.backend as backend

cpu_mask = torch.tensor([True, False], dtype=torch.bool)
meta_mask = torch.ones(2, dtype=torch.bool, device="meta")

logical_result = backend.logical_and(cpu_mask, meta_mask)

assert logical_result.device.type == "meta"
assert logical_result.dtype == torch.bool
assert logical_result.shape == torch.Size([2])

condition = torch.tensor([True, False], dtype=torch.bool)
cpu_values = torch.tensor([1.0, 2.0])
meta_values = torch.ones(2, device="meta")

selected = backend.where(condition, cpu_values, meta_values)

assert selected.device.type == "meta"
assert selected.dtype == torch.float32
assert selected.shape == torch.Size([2])
""",
    )

    assert result.returncode == 0, result.stderr
