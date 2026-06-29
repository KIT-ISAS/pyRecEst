import pytest


@pytest.mark.backend_portable
def test_pytorch_allclose_accepts_equal_nan_after_backend_support_patch():
    pytest.importorskip("torch")

    import pyrecest._backend.pytorch as pytorch_backend
    from pyrecest.backend_support._torch_dtype_promotion_contract import (
        patch_pytorch_dtype_promotion_contract,
    )

    patch_pytorch_dtype_promotion_contract()

    assert pytorch_backend.allclose(
        [float("nan"), 1.0],
        [float("nan"), 1.0],
        equal_nan=True,
    )
    assert not pytorch_backend.allclose(
        [float("nan"), 1.0],
        [float("nan"), 1.0],
        equal_nan=False,
    )
    assert pytorch_backend.allclose([[1.0]], [1.0], atol=0.0, rtol=0.0)
