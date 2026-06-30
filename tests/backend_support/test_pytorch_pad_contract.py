import importlib.util

import pytest

from tests.support.backend_runner import run_backend_code

pytestmark = pytest.mark.backend_portable

EXPECTED_PADDED = [
    [7, 5, 5, 8],
    [7, 1, 2, 8],
    [7, 3, 4, 8],
    [7, 6, 6, 8],
]


def test_raw_pytorch_pad_accepts_numpy_style_constant_values_after_import():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    code = f"""
import pyrecest  # noqa: F401  # triggers raw-backend compatibility patches
import torch
import pyrecest._backend.pytorch as raw_pytorch_backend

values = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64)
result = raw_pytorch_backend.pad(
    values,
    ((1, 1), (1, 1)),
    mode="constant",
    constant_values=((5, 6), (7, 8)),
)
assert result.tolist() == {EXPECTED_PADDED!r}

one_dimensional = raw_pytorch_backend.pad(
    torch.tensor([1, 2], dtype=torch.int64),
    (1, 2),
    mode="constant",
    constant_values=(9, 10),
)
assert one_dimensional.tolist() == [9, 1, 2, 10, 10]
print("ok")
"""
    result = run_backend_code("numpy", code)
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_public_pytorch_pad_accepts_numpy_style_constant_values():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    code = f"""
import pyrecest.backend as backend

result = backend.pad(
    backend.array([[1, 2], [3, 4]], dtype=backend.int64),
    ((1, 1), (1, 1)),
    mode="constant",
    constant_values=((5, 6), (7, 8)),
)
assert backend.to_numpy(result).tolist() == {EXPECTED_PADDED!r}
print("ok")
"""
    result = run_backend_code("pytorch", code)
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


@pytest.mark.parametrize("backend_name", ["numpy", "pytorch"])
def test_raw_pytorch_comparison_helpers_accept_array_like_inputs(backend_name):
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    code = """
import numpy as np
import pyrecest._backend.pytorch as pt

np.testing.assert_array_equal(
    pt.to_numpy(pt.greater([2, 1, 3], [1, 1, 4])),
    np.array([True, False, False]),
)
np.testing.assert_array_equal(
    pt.to_numpy(pt.less([2, 1, 3], [1, 2, 3])),
    np.array([False, True, False]),
)
np.testing.assert_array_equal(
    pt.to_numpy(pt.less_equal([2, 1, 3], [2, 0, 4])),
    np.array([True, False, True]),
)
np.testing.assert_array_equal(
    pt.to_numpy(pt.logical_or([0, 1, 0], [0, 0, 2])),
    np.array([False, True, True]),
)
"""

    result = run_backend_code(backend_name, code)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("backend_name", ["numpy", "pytorch"])
def test_raw_pytorch_less_equal_uses_tensor_operand_device(backend_name):
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    code = """
import torch
import pyrecest._backend.pytorch as pt
import pyrecest.backend as backend

raw_left = torch.empty(3, device="meta")
raw_result = pt.less_equal(raw_left, [1.0, 2.0, 3.0])
assert raw_result.device.type == "meta"
assert tuple(raw_result.shape) == (3,)
assert raw_result.dtype == torch.bool

if backend.__backend_name__ == "pytorch":
    public_left = torch.empty(3, device="meta")
    public_result = backend.less_equal(public_left, [1.0, 2.0, 3.0])
    assert public_result.device.type == "meta"
    assert tuple(public_result.shape) == (3,)
    assert public_result.dtype == torch.bool
"""

    result = run_backend_code(backend_name, code)
    assert result.returncode == 0, result.stderr
