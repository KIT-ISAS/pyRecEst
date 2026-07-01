import pytest


def test_jax_fft_scalar_axis_smoke():
    jnp = pytest.importorskip("jax.numpy")
    from pyrecest._backend.jax import fft

    values = jnp.arange(4.0)
    axis = jnp.array(0)
    assert fft.rfft(values, axis=axis).shape == fft.rfft(values, axis=0).shape
