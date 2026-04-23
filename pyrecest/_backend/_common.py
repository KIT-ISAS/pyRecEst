import math as _math

from numpy import pi


def comb(n, k):
    return _math.factorial(n) // _math.factorial(k) // _math.factorial(n - k)


def size(x, axis=None):
    """Return the total number of elements or the length of a given axis."""
    if hasattr(x, "numel"):
        if axis is None:
            return x.numel()
        return x.shape[axis]

    shape = getattr(x, "shape", None)
    if shape is None:
        if axis is not None:
            raise ValueError("axis is only supported for array-like inputs")
        return 1

    if axis is not None:
        return shape[axis]

    result = 1
    for dim in shape:
        result *= dim
    return result
