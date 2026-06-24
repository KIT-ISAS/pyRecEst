import numpy as np
import pytest

pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
from pyrecest._backend.jax import random  # noqa: E402


def test_uniform_rejects_unusable_bounds():
    too_large = np.finfo(float).max * 2.0

    with pytest.raises(ValueError, match="finite"):
        random.uniform(0.0, too_large)

    with pytest.raises(ValueError, match="finite"):
        random.uniform(-too_large, 1.0)

    with pytest.raises(ValueError, match="finite"):
        random.uniform(jnp.array([0.0, too_large]), jnp.array([1.0, 2.0]))
