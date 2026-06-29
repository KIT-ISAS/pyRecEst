import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code


@pytest.mark.backend_portable
def test_pytorch_concatenate_axis_none_flattens_public_backend_inputs():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        "pytorch",
        """
import pyrecest.backend as backend

actual = backend.concatenate(([[1, 2], [3, 4]], [[5, 6]]), axis=None)

assert tuple(actual.shape) == (6,)
assert actual.tolist() == [1, 2, 3, 4, 5, 6]
print("ok")
""",
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


@pytest.mark.backend_portable
def test_raw_pytorch_concatenate_axis_none_is_patched_with_numpy_backend():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        "numpy",
        """
import pyrecest._backend.pytorch as pytorch_backend

actual = pytorch_backend.concatenate(([[1, 2], [3, 4]], [[5, 6]]), axis=None)

assert tuple(actual.shape) == (6,)
assert actual.tolist() == [1, 2, 3, 4, 5, 6]
print("ok")
""",
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
