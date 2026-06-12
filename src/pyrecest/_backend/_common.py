import math as _math
import os as _os

import numpy as _np
from numpy import pi


def comb(n, k):
    return _math.comb(n, k)


def outer(a, b):
    """Return a batched outer product for array/tensor backends."""
    torch_pair = _torch_promoted_pair(a, b)
    if torch_pair is not None:
        a, b = torch_pair
        if a.ndim == 0 or b.ndim == 0:
            return a * b
        a_expanded = a[..., :, None]
        b_expanded = b[..., None, :]
        return a_expanded * b_expanded

    a = _np.asarray(a)
    b = _np.asarray(b)
    if a.ndim == 0 or b.ndim == 0:
        return _np.multiply(a, b)
    a_expanded = a[..., :, None]
    b_expanded = b[..., None, :]
    return a_expanded * b_expanded


def size(x, axis=None):
    """Return the total number of elements or the length of a given axis."""
    if hasattr(x, "numel"):
        if axis is None:
            return x.numel()
        return x.shape[axis]

    shape = getattr(x, "shape", None)
    if shape is None:
        shape = _np.shape(x)

    if axis is not None:
        return shape[axis]

    result = 1
    for dim in shape:
        result *= dim
    return result


def ndim(x):
    """Return the number of dimensions for arrays and array-like inputs."""
    ndim_value = getattr(x, "ndim", None)
    if ndim_value is not None:
        return ndim_value

    return _np.ndim(x)


def _torch_module_for_values(*values):
    try:
        import torch as _torch
    except ModuleNotFoundError:
        return None

    if _os.environ.get("PYRECEST_BACKEND") == "pytorch" or any(
        _torch.is_tensor(value) for value in values
    ):
        return _torch
    return None


def _torch_promoted_pair(first, second):
    torch = _torch_module_for_values(first, second)
    if torch is None:
        return None

    device = next(
        (value.device for value in (first, second) if torch.is_tensor(value)),
        None,
    )
    first_tensor = torch.as_tensor(first, device=device)
    second_tensor = torch.as_tensor(second, device=device)
    dtype = torch.promote_types(first_tensor.dtype, second_tensor.dtype)
    return first_tensor.to(dtype=dtype), second_tensor.to(dtype=dtype)


def dot(a, b):
    torch_pair = _torch_promoted_pair(a, b)
    if torch_pair is not None:
        a, b = torch_pair
        torch = _torch_module_for_values(a, b)
        if a.ndim == 0 or b.ndim == 0:
            return torch.multiply(a, b)
        if b.ndim == 1:
            return torch.einsum("...i,i->...", a, b)
        if a.ndim == 1:
            return torch.einsum("i,...i->...", a, b)
        return torch.einsum("...i,...i->...", a, b)

    a = _np.asarray(a)
    b = _np.asarray(b)
    if a.ndim == 0 or b.ndim == 0:
        return _np.multiply(a, b)
    if b.ndim == 1:
        return _np.einsum("...i,i->...", a, b)
    if a.ndim == 1:
        return _np.einsum("i,...i->...", a, b)
    return _np.einsum("...i,...i->...", a, b)


def matvec(matrix, vector):
    torch_pair = _torch_promoted_pair(matrix, vector)
    if torch_pair is not None:
        matrix, vector = torch_pair
        torch = _torch_module_for_values(matrix, vector)
        if vector.ndim == 1:
            return torch.matmul(matrix, vector)
        if matrix.ndim == 2:
            return torch.matmul(matrix, vector.T).T
        return torch.einsum("...ij,...j->...i", matrix, vector)

    matrix = _np.asarray(matrix)
    vector = _np.asarray(vector)
    if vector.ndim == 1:
        return _np.matmul(matrix, vector)
    if matrix.ndim == 2:
        return _np.matmul(matrix, vector.T).T
    return _np.einsum("...ij,...j->...i", matrix, vector)
