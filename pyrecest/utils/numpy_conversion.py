"""Utilities for converting backend arrays to plain numpy arrays."""

import numpy as np


def to_numpy(x, dtype=None):
    """Convert to a plain numpy array, handling torch tensors.

    This avoids numpy 2.0 deprecation warnings about ``__array__`` not accepting
    the ``copy`` keyword argument, and ``__array_wrap__`` signature changes, which
    occur when torch tensors are passed directly to scipy or numpy functions.

    Parameters
    ----------
    x : array-like
        Input array.  If it is a torch tensor (i.e. has a ``detach`` method), it
        is converted via ``.detach().numpy()``.  Otherwise ``np.asarray`` is used.
    dtype : dtype-like, optional
        If provided, the result is cast to this dtype.

    Returns
    -------
    np.ndarray
    """
    if hasattr(x, "detach"):
        arr = x.detach().numpy()
        return arr.astype(dtype) if dtype is not None else arr
    return np.asarray(x, dtype=dtype)
