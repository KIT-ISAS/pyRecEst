import operator as _operator

from ._dispatch import _common
from ._dispatch import numpy as _np

_modify_func_default_dtype = _common._modify_func_default_dtype
_allow_complex_dtype = _common._allow_complex_dtype
_BOOLEAN_TYPES = (bool, _np.bool_)


def _rand(*dims, size=None):
    """Draw uniform samples with NumPy-style positional and size arguments."""
    if dims:
        if size is not None:
            raise TypeError("Specify either positional dimensions or size, not both.")
        size = dims[0] if len(dims) == 1 else dims
    return _np.random.random(size)


rand = _modify_func_default_dtype(
    copy=False, kw_only=True, target=_allow_complex_dtype(target=_rand)
)


def _contains_boolean_value(value):
    if isinstance(value, _BOOLEAN_TYPES):
        return True
    try:
        values = _np.asarray(value, dtype=object).reshape(-1)
    except (TypeError, ValueError, RuntimeError):
        return False
    return any(isinstance(item, _BOOLEAN_TYPES) for item in values)


def _validate_uniform_bound(bound, name):
    if _contains_boolean_value(bound):
        raise TypeError(f"{name} must be real numeric, not boolean")
    bound_array = _np.asarray(bound)
    if bound_array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must be real numeric")
    if _np.any(~_np.isfinite(bound_array)):
        raise ValueError("uniform bounds must be finite")
    return bound_array


def _validate_uniform_bounds(low, high):
    low_array = _validate_uniform_bound(low, "low")
    high_array = _validate_uniform_bound(high, "high")
    if _np.any(low_array > high_array):
        raise ValueError("Upper bound must be greater than or equal to lower bound")


def _uniform(low=0.0, high=1.0, size=None):
    _validate_uniform_bounds(low, high)
    return _np.random.uniform(low, high, size)


uniform = _modify_func_default_dtype(
    copy=False, kw_only=True, target=_allow_complex_dtype(target=_uniform)
)

normal = _modify_func_default_dtype(
    copy=False, kw_only=True, target=_allow_complex_dtype(target=_np.random.normal)
)

multivariate_normal = _modify_func_default_dtype(
    copy=False,
    kw_only=True,
    target=_allow_complex_dtype(target=_np.random.multivariate_normal),
)


def _normalize_choice_axis(axis, ndim):
    if isinstance(axis, _BOOLEAN_TYPES):
        raise TypeError("axis must be an integer")
    try:
        axis = _operator.index(axis)
    except TypeError as exc:
        raise TypeError("axis must be an integer") from exc
    if axis < -ndim or axis >= ndim:
        raise ValueError(f"axis {axis} is out of bounds for array of dimension {ndim}")
    return axis % ndim


def _choice_bool(value, name):
    if isinstance(value, _BOOLEAN_TYPES):
        return bool(value)
    value_array = _np.asarray(value)
    if value_array.shape == () and value_array.dtype.kind == "b":
        return bool(value_array.item())
    raise TypeError(f"{name} must be a boolean")


def _validate_choice_population(a_array):
    if a_array.ndim != 0:
        return
    scalar = a_array.item()
    if isinstance(scalar, _BOOLEAN_TYPES):
        raise ValueError("a must be a positive integer or a non-empty array")


def _maybe_preserve_choice_order(indices, *, replace, p, shuffle, size):
    if replace or p is not None or shuffle or size is None:
        return indices

    index_array = _np.asarray(indices)
    if index_array.ndim == 0:
        return indices
    return _np.sort(index_array.reshape(-1)).reshape(index_array.shape)


def choice(a, size=None, replace=True, p=None, axis=0, shuffle=True):
    """Draw samples from an integer or array population."""
    replace = _choice_bool(replace, "replace")
    shuffle = _choice_bool(shuffle, "shuffle")
    a_array = _np.asarray(a)
    _validate_choice_population(a_array)
    if a_array.ndim == 0:
        return _maybe_preserve_choice_order(
            _np.random.choice(a, size=size, replace=replace, p=p),
            replace=replace,
            p=p,
            shuffle=shuffle,
            size=size,
        )

    axis = _normalize_choice_axis(axis, a_array.ndim)
    if a_array.ndim == 1 and axis == 0:
        indices = _np.random.choice(a_array.shape[0], size=size, replace=replace, p=p)
        indices = _maybe_preserve_choice_order(
            indices,
            replace=replace,
            p=p,
            shuffle=shuffle,
            size=size,
        )
        return _np.take(a_array, indices, axis=0)

    if p is not None:
        p = _np.asarray(p)

    indices = _np.random.choice(a_array.shape[axis], size=size, replace=replace, p=p)
    indices = _maybe_preserve_choice_order(
        indices,
        replace=replace,
        p=p,
        shuffle=shuffle,
        size=size,
    )
    return _np.take(a_array, indices, axis=axis)
