import numpy as np
import pytest


torch = pytest.importorskip("torch")

import pyrecest._backend.pytorch as pytorch_backend  # noqa: E402
from pyrecest.backend_support._torch_dtype_promotion_contract import (  # noqa: E402
    patch_pytorch_dtype_promotion_contract,
)


@pytest.mark.parametrize(
    ("axis", "expected_dims"),
    [
        (np.int64(0), (0,)),
        (np.array(1, dtype=np.int64), (1,)),
        (np.array([0, 1], dtype=np.int64), (0, 1)),
        ([np.array(0), np.int64(1)], (0, 1)),
    ],
)
def test_pytorch_flip_accepts_numpy_integer_axes(axis, expected_dims):
    patch_pytorch_dtype_promotion_contract()
    values = torch.arange(6).reshape(2, 3)

    result = pytorch_backend.flip(values, axis=axis)
    expected = torch.flip(values, dims=expected_dims)

    assert torch.equal(result, expected)


@pytest.mark.parametrize("axis", [np.bool_(True), np.array(True), np.array([True])])
def test_pytorch_flip_rejects_boolean_axes(axis):
    patch_pytorch_dtype_promotion_contract()
    values = torch.arange(6).reshape(2, 3)

    with pytest.raises(TypeError, match="axis must be an integer"):
        pytorch_backend.flip(values, axis=axis)
