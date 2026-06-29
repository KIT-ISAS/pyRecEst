import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code


@pytest.mark.backend_portable
@pytest.mark.parametrize("backend_name", ["pytorch", "numpy"])
def test_pytorch_arctan_accepts_array_like_inputs(backend_name):
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    if backend_name == "pytorch":
        code = """
import pyrecest.backend as backend

actual = backend.arctan([0.0, 1.0])
expected = backend.array([0.0, 0.7853981633974483])

assert tuple(actual.shape) == (2,)
assert backend.allclose(actual, expected)
print("ok")
"""
    else:
        code = """
import pyrecest._backend.pytorch as raw_pytorch

actual = raw_pytorch.arctan([0.0, 1.0])
expected = raw_pytorch.array([0.0, 0.7853981633974483])

assert tuple(actual.shape) == (2,)
assert raw_pytorch.allclose(actual, expected)
print("ok")
"""

    result = run_backend_code(backend_name, code)

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
