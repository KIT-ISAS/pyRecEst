import pytest
from tests.support.backend_runner import run_backend_code


def test_pytorch_assignment_accepts_numpy_index_arrays():
    pytest.importorskip("torch")

    code = """
import numpy as np
import pyrecest.backend as backend
import pyrecest._backend.pytorch as raw_pytorch

assert backend.__backend_name__ == "pytorch"

indices = np.array([0, 2])
mask = np.array([True, False, True, False])
base = backend.zeros(4)

updated = backend.assignment(base, [1.0, 2.0], indices)
summed = backend.assignment_by_sum(base, [1.0, 2.0], indices)
masked = backend.assignment(base, [3.0, 4.0], mask)
masked_sum = backend.assignment_by_sum(base, [3.0, 4.0], mask)
raw_updated = raw_pytorch.assignment(raw_pytorch.zeros(4), [5.0, 6.0], indices)
raw_masked = raw_pytorch.assignment(raw_pytorch.zeros(4), [7.0, 8.0], mask)

assert backend.to_numpy(updated).tolist() == [1.0, 0.0, 2.0, 0.0]
assert backend.to_numpy(summed).tolist() == [1.0, 0.0, 2.0, 0.0]
assert backend.to_numpy(masked).tolist() == [3.0, 0.0, 4.0, 0.0]
assert backend.to_numpy(masked_sum).tolist() == [3.0, 0.0, 4.0, 0.0]
assert raw_pytorch.to_numpy(raw_updated).tolist() == [5.0, 0.0, 6.0, 0.0]
assert raw_pytorch.to_numpy(raw_masked).tolist() == [7.0, 0.0, 8.0, 0.0]
"""

    result = run_backend_code("pytorch", code)

    assert result.returncode == 0, result.stderr
