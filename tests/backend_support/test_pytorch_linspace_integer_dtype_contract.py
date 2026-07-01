import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code


@pytest.mark.backend_portable
def test_pytorch_linspace_integer_dtype_floors_negative_fractional_values():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        "pytorch",
        """
import numpy as np
import pyrecest.backend as backend

actual = backend.to_numpy(backend.linspace(-3, 3, num=5, dtype=backend.int64))
expected = np.linspace(-3, 3, num=5, dtype=np.int64)

assert actual.dtype == expected.dtype
assert actual.tolist() == expected.tolist()
""",
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.backend_portable
def test_raw_pytorch_linspace_integer_dtype_floors_negative_fractional_values():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        "numpy",
        """
import numpy as np
import pyrecest._backend.pytorch as raw_pytorch

actual = raw_pytorch.to_numpy(raw_pytorch.linspace(-3, 3, num=5, dtype=raw_pytorch.int64))
expected = np.linspace(-3, 3, num=5, dtype=np.int64)

assert actual.dtype == expected.dtype
assert actual.tolist() == expected.tolist()
""",
    )

    assert result.returncode == 0, result.stderr
