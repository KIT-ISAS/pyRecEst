import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code


@pytest.mark.backend_portable
def test_pytorch_array_and_cast_accept_torch_dtype_string_aliases():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        "pytorch",
        """
import pyrecest.backend as backend

values = backend.array([1, 2, 3], dtype="torch.float64")
assert values.dtype == backend.float64
assert backend.to_numpy(values).tolist() == [1.0, 2.0, 3.0]

cast_values = backend.cast(values, "torch.float32")
assert cast_values.dtype == backend.float32
assert backend.to_numpy(cast_values).tolist() == [1.0, 2.0, 3.0]

complex_values = backend.array([1, 2], dtype="torch.complex128")
assert complex_values.dtype == backend.complex128
assert tuple(complex_values.shape) == (2,)

print("ok")
""",
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
