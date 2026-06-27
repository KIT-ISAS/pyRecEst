import numpy as np
import pytest

pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
from pyrecest._backend.jax import random  # noqa: E402


@pytest.mark.parametrize(
    "scale",
    [
        False,
        np.bool_(True),
        jnp.array([True, False]),
        [1.0, False],
        np.array([1.0, np.bool_(False)], dtype=object),
    ],
)
def test_normal_rejects_boolean_scale(scale):
    with pytest.raises(TypeError, match="scale must be real numeric, not boolean"):
        random.normal(scale=scale)
