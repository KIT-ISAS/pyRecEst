import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code

pytorch_backend = pytest.importorskip("pyrecest._backend.pytorch")


def test_pytorch_allclose_accepts_mixed_boolean_numeric_inputs():
    left = pytorch_backend.array([True, False])
    right = pytorch_backend.array([1, 0], dtype=pytorch_backend.uint8)

    assert bool(pytorch_backend.allclose(left, right))


def test_pytorch_isclose_accepts_mixed_boolean_numeric_inputs():
    left = pytorch_backend.array([True, False])
    right = pytorch_backend.array([1.0, 0.0], dtype=pytorch_backend.float32)

    assert pytorch_backend.isclose(left, right).tolist() == [True, True]


@pytest.mark.backend_portable
def test_public_pytorch_convert_to_wider_dtype_uses_torch_promotion_contract():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        "pytorch",
        "import pyrecest.backend as backend\n"
        "left = backend.array([True, False])\n"
        "right = backend.array([1, 0], dtype=backend.uint8)\n"
        "promoted_left, promoted_right = backend.convert_to_wider_dtype([left, right])\n"
        "assert promoted_left.dtype == backend.uint8, promoted_left.dtype\n"
        "assert promoted_right.dtype == backend.uint8, promoted_right.dtype\n"
        "assert backend.to_numpy(promoted_left).tolist() == [1, 0]\n",
    )

    assert result.returncode == 0, result.stderr
