import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
from pyrecest._backend.jax import random  # noqa: E402


@pytest.mark.parametrize("bounds_type", [list, tuple])
def test_randint_accepts_sequence_bounds_with_explicit_size(bounds_type):
    random.seed(0)
    low = bounds_type([0, 10])
    high = bounds_type([3, 13])

    samples = random.randint(low, high, size=(4, 2))

    assert samples.shape == (4, 2)
    assert jnp.all(samples >= jnp.array([0, 10]))
    assert jnp.all(samples < jnp.array([3, 13]))


@pytest.mark.parametrize(
    ("low", "high"),
    [
        (3, 3),
        (4, 1),
        (jnp.array([0, 10]), jnp.array([3, 10])),
    ],
)
def test_randint_rejects_non_increasing_bounds(low, high):
    with pytest.raises(ValueError, match="high must be greater than low"):
        random.randint(low, high, size=(2,))
