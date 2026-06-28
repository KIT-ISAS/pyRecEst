import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")
from pyrecest._backend import jax as jax_backend  # noqa: E402


def _to_numpy(value):
    return np.asarray(jax_backend.to_numpy(value))


def test_direct_jax_trace_accepts_explicit_axes_offset_dtype_and_out():
    values = jnp.array(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
        ],
        dtype=jnp.float32,
    )
    out = jnp.zeros((2,), dtype=jnp.float32)

    result = jax_backend.trace(
        values,
        offset=1,
        axis1=1,
        axis2=2,
        dtype=jnp.float32,
        out=out,
    )

    assert _to_numpy(result).tolist() == [8.0, 20.0]


def test_direct_jax_trace_defaults_to_last_two_axes():
    values = jnp.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ]
    )

    assert _to_numpy(jax_backend.trace(values)).tolist() == [5.0, 13.0]
