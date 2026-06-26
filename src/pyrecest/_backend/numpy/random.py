"""Numpy based random backend."""

import numpy as _np
from numpy.random import default_rng as _default_rng
from numpy.random import (  # For PyRecEst
    get_state,
    multinomial,
    seed,
    set_state,
)

from .._shared_numpy.random import choice, multivariate_normal, normal, rand, uniform

_BOOLEAN_TYPES = (bool, _np.bool_)


def _contains_boolean_value(value):
    if isinstance(value, _BOOLEAN_TYPES):
        return True
    try:
        values = _np.asarray(value, dtype=object).reshape(-1)
    except (TypeError, ValueError, RuntimeError):
        return False
    return any(isinstance(item, _BOOLEAN_TYPES) for item in values)


def _validate_randint_bound(bound, name):
    if _contains_boolean_value(bound):
        raise TypeError(f"{name} must contain integer values")
    try:
        bound_array = _np.asarray(bound)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain integer values") from exc
    if bound_array.dtype.kind not in "iu":
        raise TypeError(f"{name} must contain integer values")


def randint(low, high=None, size=None, dtype=int):
    """Draw integer samples after rejecting non-integer bounds."""

    if high is None:
        _validate_randint_bound(low, "high")
        return _np.random.randint(low, high=None, size=size, dtype=dtype)

    _validate_randint_bound(low, "low")
    _validate_randint_bound(high, "high")
    return _np.random.randint(low, high=high, size=size, dtype=dtype)
