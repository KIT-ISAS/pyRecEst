import math as _math

import numpy as _np
from numpy import pi


def comb(n, k):
    return _math.comb(n, k)


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
