"""JAX signal backend wrappers."""

import jax.numpy as _jnp
from jax.scipy import signal as _signal


def fftconvolve(in1, in2, mode="full", axes=None):
    """Convolve array-like inputs with JAX arrays before dispatch."""
    return _signal.fftconvolve(
        _jnp.asarray(in1),
        _jnp.asarray(in2),
        mode=mode,
        axes=axes,
    )
