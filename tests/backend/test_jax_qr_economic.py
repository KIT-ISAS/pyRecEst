import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")
from pyrecest._backend.jax import linalg as jax_linalg  # noqa: E402


def test_jax_linalg_qr_accepts_numpy_economic_mode():
    values = jnp.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]
    )

    result = jax_linalg.qr(values, mode="economic")

    assert result.shape == values.shape
    assert np.isfinite(np.asarray(result)).all()
