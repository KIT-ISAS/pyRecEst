import pytest

jnp = pytest.importorskip("jax.numpy")


def test_placeholder():
    assert jnp is not None
