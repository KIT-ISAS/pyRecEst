import importlib.util

import pytest

from tests.support.backend_runner import run_backend_code

pytestmark = pytest.mark.backend_portable


def test_raw_pytorch_creation_helpers_reject_boolean_shapes_after_import():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    code = """
import numpy as np
import torch
import pyrecest  # noqa: F401  # triggers raw-backend compatibility patches
import pyrecest._backend.pytorch as raw_pytorch_backend

invalid_shapes = [
    True,
    False,
    np.bool_(True),
    [True],
    (True, 2),
    np.array(True),
    np.array([True]),
    torch.tensor(True),
    torch.tensor([True]),
]

for helper_name in ("empty", "zeros", "ones"):
    helper = getattr(raw_pytorch_backend, helper_name)
    for shape in invalid_shapes:
        try:
            helper(shape)
        except TypeError:
            pass
        else:
            raise AssertionError(f"{helper_name} accepted boolean shape {shape!r}")

for shape in invalid_shapes:
    try:
        raw_pytorch_backend.full(shape, 1)
    except TypeError:
        pass
    else:
        raise AssertionError(f"full accepted boolean shape {shape!r}")

assert tuple(raw_pytorch_backend.zeros((2,)).shape) == (2,)
print("ok")
"""
    result = run_backend_code("numpy", code)
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_public_pytorch_creation_helpers_reject_boolean_shapes():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    code = """
import numpy as np
import torch
import pyrecest.backend as backend

invalid_shapes = [
    True,
    False,
    np.bool_(True),
    [True],
    (True, 2),
    np.array(True),
    np.array([True]),
    torch.tensor(True),
    torch.tensor([True]),
]

for helper_name in ("empty", "zeros", "ones"):
    helper = getattr(backend, helper_name)
    for shape in invalid_shapes:
        try:
            helper(shape)
        except TypeError:
            pass
        else:
            raise AssertionError(f"{helper_name} accepted boolean shape {shape!r}")

for shape in invalid_shapes:
    try:
        backend.full(shape, 1)
    except TypeError:
        pass
    else:
        raise AssertionError(f"full accepted boolean shape {shape!r}")

assert tuple(backend.zeros((2,)).shape) == (2,)
print("ok")
"""
    result = run_backend_code("pytorch", code)
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
