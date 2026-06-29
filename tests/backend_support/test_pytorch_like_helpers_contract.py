import pytest

import pyrecest.backend as backend
import pyrecest.backend_support  # noqa: F401  ensure compatibility patches are installed

pytorch_backend = pytest.importorskip("pyrecest._backend.pytorch")


def test_raw_pytorch_like_helpers_accept_array_like_inputs():
    assert pytorch_backend.to_numpy(pytorch_backend.zeros_like([1, 2, 3])).tolist() == [
        0,
        0,
        0,
    ]
    assert pytorch_backend.to_numpy(pytorch_backend.ones_like([1, 2, 3])).tolist() == [
        1,
        1,
        1,
    ]
    assert pytorch_backend.to_numpy(pytorch_backend.full_like([1, 2, 3], 7)).tolist() == [
        7,
        7,
        7,
    ]

    empty_result = pytorch_backend.empty_like([[1.0, 2.0], [3.0, 4.0]])
    assert tuple(empty_result.shape) == (2, 2)


def test_public_pytorch_like_helpers_accept_array_like_inputs():
    if getattr(backend, "__backend_name__", None) != "pytorch":
        pytest.skip("public PyTorch backend is not active")

    assert backend.to_numpy(backend.zeros_like([1, 2, 3])).tolist() == [0, 0, 0]
    assert backend.to_numpy(backend.ones_like([1, 2, 3])).tolist() == [1, 1, 1]
    assert backend.to_numpy(backend.full_like([1, 2, 3], 7)).tolist() == [7, 7, 7]

    empty_result = backend.empty_like([[1.0, 2.0], [3.0, 4.0]])
    assert tuple(empty_result.shape) == (2, 2)
