import pytest

import pyrecest.backend as backend


def _to_python(value):
    value = backend.to_numpy(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def test_public_pytorch_one_hot_accepts_scalar_label():
    if backend.__backend_name__ != "pytorch":
        pytest.skip("PyTorch-specific one_hot scalar-label contract")

    result = backend.one_hot(1, 3)

    assert result.shape == (3,)
    assert str(backend.to_numpy(result).dtype) == "uint8"
    assert _to_python(result) == [0, 1, 0]


def test_raw_pytorch_one_hot_accepts_scalar_label_after_package_import():
    if backend.__backend_name__ != "pytorch":
        pytest.skip("PyTorch-specific raw-backend one_hot scalar-label contract")

    import pyrecest._backend.pytorch as pytorch_backend  # pylint: disable=import-outside-toplevel

    result = pytorch_backend.one_hot(1, 3)

    assert result.shape == (3,)
    assert pytorch_backend.to_numpy(result).dtype.name == "uint8"
    assert pytorch_backend.to_numpy(result).tolist() == [0, 1, 0]
