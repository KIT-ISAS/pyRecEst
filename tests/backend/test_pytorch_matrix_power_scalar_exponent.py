import importlib.util

import pytest

from tests.support.backend_runner import run_backend_code


@pytest.mark.backend_portable
def test_pytorch_matrix_power_accepts_scalar_array_exponents():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    code = """
import numpy as np
import torch

import pyrecest.backend as backend
from pyrecest.backend import linalg

matrix = [[1.0, 1.0], [0.0, 1.0]]
expected = [[1.0, 2.0], [0.0, 1.0]]

assert backend.to_numpy(linalg.matrix_power(matrix, np.array(2))).tolist() == expected
assert backend.to_numpy(linalg.matrix_power(matrix, torch.tensor(2))).tolist() == expected
"""
    result = run_backend_code("pytorch", code)
    assert result.returncode == 0, result.stderr
