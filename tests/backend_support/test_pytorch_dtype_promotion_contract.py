import pytest

pytorch_backend = pytest.importorskip("pyrecest._backend.pytorch")


def test_pytorch_allclose_accepts_mixed_boolean_numeric_inputs():
    left = pytorch_backend.array([True, False])
    right = pytorch_backend.array([1, 0], dtype=pytorch_backend.uint8)

    assert bool(pytorch_backend.allclose(left, right))


def test_pytorch_isclose_accepts_mixed_boolean_numeric_inputs():
    left = pytorch_backend.array([True, False])
    right = pytorch_backend.array([1.0, 0.0], dtype=pytorch_backend.float32)

    assert pytorch_backend.isclose(left, right).tolist() == [True, True]
