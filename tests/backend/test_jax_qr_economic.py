import pytest

jnp = pytest.importorskip("jax.numpy")
from pyrecest._backend.jax import linalg as jax_linalg  # noqa: E402


def test_placeholder():
    assert jax_linalg is not None
