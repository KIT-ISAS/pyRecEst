import jax.numpy as _jnp
from pyrecest._backend._dtype_utils import (
    _pre_set_default_dtype,
    get_default_cdtype,
    get_default_dtype,
)

# Mapping of string dtype representations to JAX dtypes
MAP_DTYPE = {
    "float32": _jnp.float32,
    "float64": _jnp.float64,
    "complex64": _jnp.complex64,
    "complex128": _jnp.complex128,
}


def as_dtype(value):
    """
    Transform string representing dtype into JAX dtype.

    Parameters
    ----------
    value : str
        String representing the dtype to be converted.

    Returns
    -------
    dtype : jnp.dtype
        JAX dtype object corresponding to the input string.
    """
    return MAP_DTYPE[value]


set_default_dtype = _pre_set_default_dtype(as_dtype)
