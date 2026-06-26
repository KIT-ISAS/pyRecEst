import numpy as np
import pytest

pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
from pyrecest._backend.jax import random  # noqa: E402


def _size_aware_samplers():
    values = jnp.array([0, 1, 2])
    mean = jnp.array([0.0])
    cov = jnp.eye(1)
    return (
        lambda size: random.rand(size=size),
        lambda size: random.uniform(size=size),
        lambda size: random.normal(size=size),
        lambda size: random.randint(0, 3, size=size),
        lambda size: random.choice(values, size=size),
        lambda size: random.multivariate_normal(mean, cov, size=size),
        lambda size: random.multinomial(3, [0.25, 0.75], size=size),
    )


def test_size_arguments_accept_zero_dimensional_integer_arrays():
    random.seed(0)
    scalar_size = np.array(3, dtype=np.int64)

    for sampler in _size_aware_samplers():
        sample = sampler(scalar_size)

        assert sample.shape[0] == 3


def test_size_arguments_reject_zero_dimensional_boolean_arrays():
    for sampler in _size_aware_samplers():
        with pytest.raises(TypeError):
            sampler(np.array(True))
