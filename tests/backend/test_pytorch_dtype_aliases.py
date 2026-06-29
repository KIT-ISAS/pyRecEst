import pytest
from tests.support.backend_runner import run_backend_code


def test_pytorch_as_dtype_resolves_common_aliases():
    pytest.importorskip("torch")

    code = """
import numpy as np
import torch
import pyrecest.backend as backend

expected = {
    "bool": torch.bool,
    "uint8": torch.uint8,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "complex64": torch.complex64,
    "complex128": torch.complex128,
}

for alias, expected_dtype in expected.items():
    assert backend.as_dtype(alias) is expected_dtype, alias
    assert backend.as_dtype(np.dtype(alias)) is expected_dtype, alias

assert backend.as_dtype(torch.float32) is torch.float32
"""

    result = run_backend_code("pytorch", code)

    assert result.returncode == 0, result.stderr
