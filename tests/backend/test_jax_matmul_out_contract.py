import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")
from pyrecest._backend import jax as jax_backend  # noqa: E402


def _to_numpy(value):
    return np.asarray(jax_backend.to_numpy(value))


def test_direct_jax_matmul_out_controls_return_dtype():
    left = jnp.array([[1, 2], [3, 4]], dtype=jnp.int32)
    right = jnp.array([[5, 6], [7, 8]], dtype=jnp.int32)
    out = jnp.zeros((2, 2), dtype=jnp.float32)

    result = jax_backend.matmul(left, right, out=out)

    assert _to_numpy(result).dtype == np.float32
    assert _to_numpy(result).tolist() == [[19.0, 22.0], [43.0, 50.0]]


def test_direct_jax_matmul_out_validates_shape():
    left = jnp.array([[1, 2], [3, 4]], dtype=jnp.int32)
    right = jnp.array([[5, 6], [7, 8]], dtype=jnp.int32)
    out = jnp.zeros((3, 3), dtype=jnp.float32)

    with pytest.raises(ValueError, match="Incompatible shapes|broadcast"):
        jax_backend.matmul(left, right, out=out)
