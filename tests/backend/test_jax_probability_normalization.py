import numpy as np
import pytest

pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
from pyrecest._backend.jax import random  # noqa: E402


def test_choice_accepts_tiny_probability_weights():
    random.seed(0)

    samples = random.choice([10, 20], size=8, p=[1e-300, 2e-300])

    assert samples.shape == (8,)
    assert set(np.asarray(samples).tolist()).issubset({10, 20})


def test_multinomial_accepts_tiny_probability_weights():
    random.seed(0)

    samples = random.multinomial(5, [1e-300, 2e-300], size=3)

    assert samples.shape == (3, 2)
    assert jnp.all(jnp.sum(samples, axis=-1) == 5)
