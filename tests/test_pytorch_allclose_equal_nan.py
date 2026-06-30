import pytest

from tests.support.backend_runner import run_backend_code


def test_pytorch_allclose_accepts_numpy_equal_nan_keyword():
    pytest.importorskip("torch")
    code = """
import pyrecest
import pyrecest.backend as backend
import pyrecest._backend.pytorch as raw_pytorch

assert bool(backend.allclose([float("nan")], [float("nan")], equal_nan=True))
assert not bool(backend.allclose([float("nan")], [float("nan")], equal_nan=False))
assert bool(raw_pytorch.allclose([float("nan")], [float("nan")], equal_nan=True))
"""

    result = run_backend_code("pytorch", code)

    assert result.returncode == 0, result.stderr
