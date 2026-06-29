import pytest

try:
    from pyrecest._backend import pytorch as pytorch_backend
except ModuleNotFoundError:
    pytorch_backend = None


@pytest.mark.skipif(pytorch_backend is None, reason="PyTorch is not installed")
def test_pytorch_linalg_pd_check_uses_backend_tolerance():
    value = pytorch_backend.array(
        [[1.0, 5e-7], [0.0, 1.0]], dtype=pytorch_backend.float32
    )

    assert pytorch_backend.linalg.is_single_matrix_pd(value)
