import importlib.util

import pytest

from tests.support.backend_runner import run_backend_code

pytestmark = pytest.mark.backend_portable


def test_raw_pytorch_flip_accepts_numpy_style_axes_after_import():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    code = """
import numpy as np
import pyrecest  # noqa: F401  # triggers raw-backend compatibility patches
import pyrecest._backend.pytorch as raw_pytorch_backend

values = np.arange(24).reshape(2, 3, 4)
axes = [np.array([2, 0]), np.array(1), np.int64(-1)]

for axis in axes:
    result = raw_pytorch_backend.flip(values, axis=axis)
    expected = np.flip(values, axis=axis)
    assert result.tolist() == expected.tolist()
print("ok")
"""
    result = run_backend_code("numpy", code)
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_public_pytorch_flip_accepts_numpy_style_axes():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    code = """
import numpy as np
import pyrecest.backend as backend

values = np.arange(24).reshape(2, 3, 4)
axes = [np.array([2, 0]), np.array(1), np.int64(-1)]

for axis in axes:
    result = backend.flip(backend.array(values), axis=axis)
    expected = np.flip(values, axis=axis)
    assert backend.to_numpy(result).tolist() == expected.tolist()
print("ok")
"""
    result = run_backend_code("pytorch", code)
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
