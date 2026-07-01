import importlib.util

import pytest

from tests.support.backend_runner import run_backend_code

pytestmark = pytest.mark.backend_portable


@pytest.mark.parametrize("backend_name", ["numpy", "autograd"])
def test_assignment_by_sum_accumulates_duplicate_advanced_indices(backend_name):
    if backend_name == "autograd" and importlib.util.find_spec("autograd") is None:
        pytest.skip("autograd is not installed")

    code = """
import pyrecest.backend as backend

vector = backend.zeros(3)
vector_result = backend.assignment_by_sum(vector, [1.0, 2.0], [0, 0])
assert backend.to_numpy(vector_result).tolist() == [3.0, 0.0, 0.0]

matrix = backend.zeros((2, 2))
matrix_result = backend.assignment_by_sum(matrix, [1.0, 2.0], [(0, 1), (0, 1)])
assert backend.to_numpy(matrix_result).tolist() == [[0.0, 3.0], [0.0, 0.0]]

print("ok")
"""
    result = run_backend_code(backend_name, code)

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
