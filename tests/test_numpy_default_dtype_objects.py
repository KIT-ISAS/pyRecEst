import numpy as np
import pytest


def _dtype_name(value):
    name = getattr(value, "name", None) or getattr(value, "__name__", None)
    if name is not None:
        return str(name)
    text = str(value)
    if "." in text:
        text = text.rsplit(".", maxsplit=1)[-1]
    return text.strip("'>")


def test_numpy_backend_set_default_dtype_accepts_dtype_like_values():
    from pyrecest.backends import get_backend

    numpy_backend = get_backend("numpy")
    previous_dtype = numpy_backend.get_default_dtype()

    try:
        result = numpy_backend.set_default_dtype(np.dtype("float32"))
        assert result == np.dtype("float32")
        assert numpy_backend.get_default_dtype() == np.dtype("float32")
        assert numpy_backend.get_default_cdtype() == np.dtype("complex64")

        result = numpy_backend.set_default_dtype(np.float64)
        assert result == np.dtype("float64")
        assert numpy_backend.get_default_dtype() == np.dtype("float64")
        assert numpy_backend.get_default_cdtype() == np.dtype("complex128")
    finally:
        numpy_backend.set_default_dtype(_dtype_name(previous_dtype))


def test_numpy_backend_set_default_dtype_rejects_nonfloating_dtype_without_mutation():
    from pyrecest.backends import get_backend

    numpy_backend = get_backend("numpy")
    previous_dtype = numpy_backend.get_default_dtype()
    previous_cdtype = numpy_backend.get_default_cdtype()

    try:
        with pytest.raises(ValueError, match="float32 or float64"):
            numpy_backend.set_default_dtype(np.dtype("complex64"))

        assert numpy_backend.get_default_dtype() == previous_dtype
        assert numpy_backend.get_default_cdtype() == previous_cdtype
    finally:
        numpy_backend.set_default_dtype(_dtype_name(previous_dtype))
