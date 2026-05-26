import pyrecest.backend as backend
import pytest


def _to_python(value):
    value = backend.to_numpy(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def test_matmul_accepts_array_like_matrix_operands():
    result = backend.matmul([[1, 2], [3, 4]], [[5], [6]])

    assert _to_python(result) == [[17], [39]]


def test_matmul_accepts_array_like_operands_with_out():
    out = backend.empty((2, 1), dtype=backend.float64)

    result = backend.matmul([[1.0, 2.0], [3.0, 4.0]], [[5.0], [6.0]], out=out)

    assert result is out
    assert _to_python(out) == [[17.0], [39.0]]


def test_matmul_rejects_array_like_vector_operands_explicitly():
    with pytest.raises(ValueError, match="ndims must be >=2"):
        backend.matmul([1, 2], [[3], [4]])
