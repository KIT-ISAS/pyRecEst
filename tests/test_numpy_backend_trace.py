import pyrecest.backend as backend
import pytest
from pyrecest.backend import array


def _to_python(value):
    value = backend.to_numpy(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def test_numpy_backend_trace_accepts_explicit_axes_offset_dtype_and_out():
    if backend.__backend_name__ != "numpy":
        pytest.skip("NumPy backend trace regression test")

    values = array(
        [
            [[1, 2, 3], [4, 5, 6]],
            [[7, 8, 9], [10, 11, 12]],
        ]
    )
    out = backend.zeros((2,), dtype=backend.float64)

    result = backend.trace(
        values,
        offset=1,
        axis1=1,
        axis2=2,
        dtype=backend.float64,
        out=out,
    )

    assert result is out
    assert _to_python(result) == [8.0, 20.0]


def test_backend_trace_defaults_to_last_two_axes():
    values = array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ]
    )

    assert _to_python(backend.trace(values)) == [5.0, 13.0]
