# For ffts. Added for pyrecest.
from functools import wraps as _wraps

import torch as _torch

from ._common import array as _array


def _as_fft_tensor(value):
    """Convert array-like FFT inputs to torch tensors."""
    return value if _torch.is_tensor(value) else _array(value)


def _wrap_arraylike_fft(torch_func):
    """Return a PyRecEst-compatible FFT helper accepting array-like input."""

    @_wraps(torch_func)
    def fft_func(input, *args, **kwargs):  # pylint: disable=redefined-builtin
        return torch_func(_as_fft_tensor(input), *args, **kwargs)

    return fft_func


rfft = _wrap_arraylike_fft(_torch.fft.rfft)
irfft = _wrap_arraylike_fft(_torch.fft.irfft)
fftshift = _wrap_arraylike_fft(_torch.fft.fftshift)
ifftshift = _wrap_arraylike_fft(_torch.fft.ifftshift)
fftn = _wrap_arraylike_fft(_torch.fft.fftn)
ifftn = _wrap_arraylike_fft(_torch.fft.ifftn)
