import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code


@pytest.mark.backend_portable
def test_pytorch_stack_helpers_accept_array_like_sequences():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    result = run_backend_code(
        "pytorch",
        """
import pyrecest.backend as backend
import pyrecest._backend.pytorch as raw_backend

assert backend.to_numpy(backend.stack(([1, 2], [3, 4]), axis=1)).tolist() == [[1, 3], [2, 4]]
assert backend.to_numpy(backend.stack(([1, 2], [3, 4]), dim=1)).tolist() == [[1, 3], [2, 4]]
assert backend.to_numpy(backend.concatenate(([1, 2], [3, 4]), axis=0)).tolist() == [1, 2, 3, 4]
assert backend.to_numpy(backend.concatenate(([[1], [2]], [[3], [4]]), axis=1)).tolist() == [[1, 3], [2, 4]]
assert backend.to_numpy(raw_backend.stack(([1, 2], [3, 4]), axis=1)).tolist() == [[1, 3], [2, 4]]
assert backend.to_numpy(raw_backend.concatenate(([1, 2], [3, 4]), axis=0)).tolist() == [1, 2, 3, 4]

assert backend.to_numpy(backend.hstack(([1, 2], [3, 4]))).tolist() == [1, 2, 3, 4]
assert backend.to_numpy(backend.vstack(([1, 2], [3, 4]))).tolist() == [[1, 2], [3, 4]]
assert backend.to_numpy(backend.column_stack(([1, 2], [3, 4]))).tolist() == [[1, 3], [2, 4]]
assert backend.to_numpy(backend.dstack(([1, 2], [3, 4]))).tolist() == [[[1, 3], [2, 4]]]

values = backend.array([[1, 2], [3, 4]])
assert backend.to_numpy(backend.hstack((values, [[5, 6], [7, 8]]))).tolist() == [[1, 2, 5, 6], [3, 4, 7, 8]]
assert backend.to_numpy(backend.column_stack((values, [[5, 6], [7, 8]]))).tolist() == [[1, 2, 5, 6], [3, 4, 7, 8]]
print("ok")
""",
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
