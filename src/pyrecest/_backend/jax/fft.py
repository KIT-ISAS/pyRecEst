"""JAX FFT backend wrappers."""

import jax.numpy as _jnp
from jax.numpy import fft as _fft


def rfft(a, n=None, axis=-1, norm=None):
    return _fft.rfft(_jnp.asarray(a), n=n, axis=axis, norm=norm)


def irfft(a, n=None, axis=-1, norm=None):
    return _fft.irfft(_jnp.asarray(a), n=n, axis=axis, norm=norm)


def fftn(a, s=None, axes=None, norm=None):
    return _fft.fftn(_jnp.asarray(a), s=s, axes=axes, norm=norm)


def ifftn(a, s=None, axes=None, norm=None):
    return _fft.ifftn(_jnp.asarray(a), s=s, axes=axes, norm=norm)


def fftshift(x, axes=None):
    return _fft.fftshift(_jnp.asarray(x), axes=axes)


def ifftshift(x, axes=None):
    return _fft.ifftshift(_jnp.asarray(x), axes=axes)
