import importlib.util

import numpy as np
import numpy.testing as npt
import pyrecest.backend as backend
import pytest


def _to_python(value):
    value = backend.to_numpy(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def test_pytorch_backend_exposes_dot_and_matvec():
    if backend.__backend_name__ != "pytorch":
        pytest.skip("PyTorch-specific backend helper contract")

    first = backend.asarray([[1.0, 2.0], [3.0, 4.0]])
    second = backend.asarray([[5.0, 6.0], [7.0, 8.0]])

    assert _to_python(backend.dot(first, second)) == [17.0, 53.0]
    assert _to_python(backend.dot(first, [5.0, 6.0])) == [17.0, 39.0]
    assert _to_python(backend.outer([1.0, 2.0], [3.0, 4.0])) == [
        [3.0, 4.0],
        [6.0, 8.0],
    ]
    assert float(_to_python(backend.dot(backend.asarray(2.0), 3.0))) == pytest.approx(
        6.0
    )

    batched_matvec = backend.matvec(
        backend.asarray([[[1.0, 0.0], [0.0, 1.0]], [[2.0, 0.0], [0.0, 3.0]]]),
        first,
    )
    shared_vector_matvec = backend.matvec(
        backend.asarray([[[1.0, 0.0], [0.0, 1.0]], [[2.0, 0.0], [0.0, 3.0]]]),
        backend.asarray([1.0, 2.0]),
    )

    assert _to_python(batched_matvec) == [[1.0, 2.0], [6.0, 12.0]]
    assert _to_python(shared_vector_matvec) == [[1.0, 2.0], [2.0, 6.0]]


def test_raw_pytorch_log1p_accepts_array_like_inputs_and_out():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    import pyrecest._backend.pytorch as pytorch_backend  # pylint: disable=import-outside-toplevel

    values = [0.0, 1.0, 3.0]
    expected = np.log1p(values)

    result = pytorch_backend.log1p(values)
    npt.assert_allclose(pytorch_backend.to_numpy(result), expected)

    out = pytorch_backend.zeros(3)
    returned = pytorch_backend.log1p(values, out=out)
    assert returned is out
    npt.assert_allclose(pytorch_backend.to_numpy(out), expected)
