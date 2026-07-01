import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code


@pytest.mark.backend_portable
def test_pytorch_moveaxis_accepts_array_like_inputs_and_numpy_axes():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        "pytorch",
        """
import numpy as np
import pyrecest.backend as backend
import pyrecest._backend.pytorch as raw_pytorch

values = np.arange(24).reshape(2, 3, 4)

public_result = backend.moveaxis(values.tolist(), np.array(0), np.array(2))
raw_result = raw_pytorch.moveaxis(values.tolist(), np.array([0, 1]), np.array([1, 2]))

assert backend.to_numpy(public_result).tolist() == np.moveaxis(values, 0, 2).tolist()
assert raw_pytorch.to_numpy(raw_result).tolist() == np.moveaxis(values, [0, 1], [1, 2]).tolist()
print("ok")
""",
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


@pytest.mark.backend_portable
def test_raw_pytorch_moveaxis_is_patched_under_default_backend():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        "numpy",
        """
import numpy as np
import pyrecest  # noqa: F401 - triggers backend support patches
import pyrecest._backend.pytorch as raw_pytorch

values = np.arange(24).reshape(2, 3, 4)
result = raw_pytorch.moveaxis(values.tolist(), np.array([0, 1]), np.array([1, 2]))

assert raw_pytorch.to_numpy(result).tolist() == np.moveaxis(values, [0, 1], [1, 2]).tolist()
print("ok")
""",
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
