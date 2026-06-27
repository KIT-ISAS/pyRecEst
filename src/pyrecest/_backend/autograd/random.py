"""Autograd based random backend."""

import autograd.numpy as _np
from autograd.numpy.random import get_state, randint, seed, set_state
from autograd.numpy.random import multinomial as _autograd_multinomial

from .._shared_numpy.random import choice, multivariate_normal, normal, rand, uniform


def _validate_multinomial_pvals(pvals):
    try:
        pvals_array = _np.asarray(pvals)
    except (TypeError, ValueError) as exc:
        raise TypeError("pvals must be real numeric") from exc
    if pvals_array.dtype.kind not in "iuf":
        raise TypeError("pvals must be real numeric")
    return pvals_array


def multinomial(n, pvals, size=None):
    pvals_array = _validate_multinomial_pvals(pvals)
    return _autograd_multinomial(n, pvals_array, size=size)
