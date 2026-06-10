import pytest

jnp = pytest.importorskip("jax.numpy")

from pyrecest._backend._dtype_utils import (  # noqa: E402
    get_default_cdtype,
    get_default_dtype,
)
from pyrecest._backend.jax import _dtype as jax_dtype  # noqa: E402


def _current_float_dtype_name():
    return "float32" if get_default_dtype() == jnp.float32 else "float64"


def test_jax_set_default_dtype_invokes_shared_dtype_setter():
    previous_name = _current_float_dtype_name()

    try:
        returned_dtype = jax_dtype.set_default_dtype("float32")

        assert returned_dtype == jnp.float32
        assert get_default_dtype() == jnp.float32
        assert get_default_cdtype() == jnp.complex64
    finally:
        jax_dtype.set_default_dtype(previous_name)
