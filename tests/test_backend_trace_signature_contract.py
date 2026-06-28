import pyrecest.backend as backend
from pyrecest.backend import array


def _to_python(value):
    value = backend.to_numpy(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def test_trace_accepts_numpy_signature_offset_axes_dtype_and_out():
    values = array(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
        ],
        dtype=backend.float32,
    )
    out = backend.zeros((2,), dtype=backend.float32)

    result = backend.trace(
        values,
        offset=1,
        axis1=1,
        axis2=2,
        dtype=backend.float32,
        out=out,
    )

    if backend.__backend_name__ != "jax":
        assert result is out
    assert _to_python(result) == [8.0, 20.0]
    assert str(backend.to_numpy(result).dtype) == "float32"
