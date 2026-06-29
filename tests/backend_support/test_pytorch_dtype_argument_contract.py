import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code


@pytest.mark.backend_portable
def test_pytorch_public_helpers_accept_numpy_dtype_aliases():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        "pytorch",
        """
import numpy as np
import pyrecest.backend as backend

values = backend.array([[1, 2, 3], [4, 5, 6]])

sum_result = backend.sum(values, axis=0, dtype=np.float64)
assert sum_result.dtype == backend.float64
assert backend.to_numpy(sum_result).tolist() == [5.0, 7.0, 9.0]

prod_out = backend.empty((3,), dtype=backend.int64)
prod_result = backend.prod(values, axis=0, dtype=np.dtype("int64"), out=prod_out)
assert prod_result is prod_out
assert prod_out.dtype == backend.int64
assert backend.to_numpy(prod_out).tolist() == [4, 10, 18]

line = backend.linspace(0, 1, num=3, dtype=np.dtype("float32"))
assert line.dtype == backend.float32
assert backend.to_numpy(line).tolist() == [0.0, 0.5, 1.0]

column = backend.to_ndarray([1, 2], 2, axis=1, dtype=np.float64)
assert column.dtype == backend.float64
assert tuple(column.shape) == (2, 1)

print("ok")
""",
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
