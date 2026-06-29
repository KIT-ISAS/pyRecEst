import pytest

pytorch_backend = pytest.importorskip("pyrecest._backend.pytorch")


def test_public_convert_to_wider_dtype_syncs_when_patch_is_reused(monkeypatch):
    import pyrecest.backend as backend
    from pyrecest.backend_support._torch_dtype_promotion_contract import (
        patch_pytorch_dtype_promotion_contract,
    )

    monkeypatch.setattr(backend, "__backend_name__", "pytorch")
    monkeypatch.setattr(backend, "convert_to_wider_dtype", None)

    patch_pytorch_dtype_promotion_contract()

    assert backend.convert_to_wider_dtype is pytorch_backend.convert_to_wider_dtype
