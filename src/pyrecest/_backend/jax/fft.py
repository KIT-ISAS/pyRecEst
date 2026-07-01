"""JAX FFT backend wrappers."""

import jax.numpy as _jnp
import numpy as _np
from jax.numpy import fft as _fft


def _is_non_bool_integer_dtype(dtype):
    """Return whether ``dtype`` is an integer dtype other than boolean."""
    return _np.issubdtype(dtype, _np.integer) and not _np.issubdtype(dtype, _np.bool_)


def _is_numpy_integer_scalar(value):
    """Return whether ``value`` is a NumPy integer scalar, excluding arrays."""
    value_type = type(value)
    return (
        value_type is not _np.ndarray
        and value_type.__module__ == "numpy"
        and hasattr(value, "dtype")
        and _is_non_bool_integer_dtype(value.dtype)
    )


def _normalize_real_fft_axis(axis):
    """Return a Python ``int`` for NumPy integer scalar-array FFT axes."""
    if type(axis) is _np.ndarray:
        if axis.size == 1 and _is_non_bool_integer_dtype(axis.dtype):
            return int(axis.item())
        return axis
    if _is_numpy_integer_scalar(axis):
        return int(axis)
    return axis


def _normalize_shift_axes(axes):
    """Return a Python ``int`` for NumPy integer scalar FFT-shift axes."""
    if _is_numpy_integer_scalar(axes):
        return int(axes)
    return axes


def rfft(a, n=None, axis=-1, norm=None):
    return _fft.rfft(_jnp.asarray(a), n=n, axis=_normalize_real_fft_axis(axis), norm=norm)


def irfft(a, n=None, axis=-1, norm=None):
    return _fft.irfft(_jnp.asarray(a), n=n, axis=_normalize_real_fft_axis(axis), norm=norm)


def fftn(a, s=None, axes=None, norm=None):
    return _fft.fftn(_jnp.asarray(a), s=s, axes=axes, norm=norm)


def ifftn(a, s=None, axes=None, norm=None):
    return _fft.ifftn(_jnp.asarray(a), s=s, axes=axes, norm=norm)


def fftshift(x, axes=None):
    return _fft.fftshift(_jnp.asarray(x), axes=_normalize_shift_axes(axes))


def ifftshift(x, axes=None):
    return _fft.ifftshift(_jnp.asarray(x), axes=_normalize_shift_axes(axes))
