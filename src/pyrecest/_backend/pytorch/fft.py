# For ffts. Added for pyrecest.
from functools import wraps as _wraps

import numpy as _np
import torch as _torch


def _as_fft_tensor(value):
    """Convert array-like FFT inputs to torch tensors."""
    if _torch.is_tensor(value):
        return value
    if isinstance(value, _np.ndarray) and any(stride < 0 for stride in value.strides):
        value = value.copy()
    return _torch.as_tensor(value)


def _resolve_fft_dim(dim, *, axis=None, axes=None):
    """Resolve NumPy-style axis aliases to PyTorch's dim argument."""
    if axis is not None:
        if axes is not None:
            raise TypeError("axis and axes cannot both be specified")
        if dim is not None:
            raise TypeError("dim and axis cannot both be specified")
        return axis
    if axes is not None:
        if dim is not None:
            raise TypeError("dim and axes cannot both be specified")
        return axes
    return dim


def _wrap_arraylike_fft(torch_func):
    """Return a PyRecEst-compatible FFT helper accepting array-like input."""

    @_wraps(torch_func)
    def fft_func(input, *args, axis=None, axes=None, dim=None, **kwargs):  # pylint: disable=redefined-builtin
        resolved_dim = _resolve_fft_dim(dim, axis=axis, axes=axes)
        if resolved_dim is not None:
            kwargs["dim"] = resolved_dim
        return torch_func(_as_fft_tensor(input), *args, **kwargs)

    return fft_func


rfft = _wrap_arraylike_fft(_torch.fft.rfft)
irfft = _wrap_arraylike_fft(_torch.fft.irfft)
fftshift = _wrap_arraylike_fft(_torch.fft.fftshift)
ifftshift = _wrap_arraylike_fft(_torch.fft.ifftshift)
fftn = _wrap_arraylike_fft(_torch.fft.fftn)
ifftn = _wrap_arraylike_fft(_torch.fft.ifftn)
