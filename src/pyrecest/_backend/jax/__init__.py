"""Jax-based computation backend.
based on implementation by Emile Mathieu
for Riemannian Score-based SDE
"""

import builtins as _builtins
import numbers as _numbers

import jax.numpy as _jnp
from jax import vmap
from jax.numpy import (  # For pyrecest; For Riemannian score-based SDE
    abs,
    all,
    allclose,
    amax,
    amin,
    angle,
    any,
    apply_along_axis,
    arange,
    arccos,
    arccosh,
    arcsin,
    arctan,
    arctan2,
    arctanh,
    argmax,
    argmin,
    argsort,
    array_equal,
    asarray,
    atleast_1d,
    atleast_2d,
    broadcast_arrays,
    broadcast_to,
    ceil,
    clip,
    column_stack,
    complex64,
    complex128,
    concatenate,
    conj,
    copy,
    cos,
    cosh,
    count_nonzero,
    cov,
    cross,
    cumprod,
    cumsum,
    deg2rad,
    diag,
    diag_indices,
    diagonal,
    diff,
    divide,
    dot,
    dstack,
    einsum,
    empty,
    empty_like,
    equal,
    exp,
    expand_dims,
    eye,
    flip,
    float32,
    float64,
    floor,
    full,
    full_like,
    greater,
    hsplit,
    hstack,
    imag,
    int32,
    int64,
    isclose,
    isfinite,
    isinf,
    isnan,
    isreal,
    kron,
    less,
    less_equal,
    linspace,
    log,
    log1p,
    logical_and,
    logical_or,
    matmul,
    max,
    maximum,
    mean,
    min,
    minimum,
    mod,
    moveaxis,
    ndim,
    nonzero,
    ones,
    ones_like,
    outer,
    pad,
    power,
    prod,
    quantile,
    rad2deg,
    real,
    repeat,
    reshape,
    roll,
    round,
    searchsorted,
    shape,
    sign,
    sin,
    sinh,
    sort,
    split,
    sqrt,
    squeeze,
    stack,
    std,
    sum,
    tan,
    tanh,
    tile,
    transpose,
    tril,
    tril_indices,
    triu,
    triu_indices,
    uint8,
    unique,
    vectorize,
    vstack,
    where,
    zeros,
    zeros_like,
)


def has_autodiff():
    """If allows for automatic differentiation.

    Returns
    -------
    has_autodiff : bool
    """
    return True


def isscalar(x):
    return _jnp.isscalar(x) and not isinstance(x, _jnp.ndarray)


def meshgrid(*xi, copy=True, sparse=False, indexing="xy"):
    """Return coordinate matrices with NumPy-style axis coercion.

    The PyRecEst backend contract follows NumPy semantics: callers may pass
    lists, ranges, and scalar axes. JAX's native ``meshgrid`` requires array or
    scalar arguments and then rejects 0-D array axes. Coercing every axis to
    at least one-dimensional JAX arrays preserves NumPy-compatible behavior while
    keeping the returned arrays in the JAX backend.
    """
    axes = tuple(_jnp.atleast_1d(_jnp.asarray(axis)) for axis in xi)
    return _jnp.meshgrid(*axes, copy=copy, sparse=sparse, indexing=indexing)


from jax import device_get as to_numpy
from jax.numpy import array
from jax.numpy import asarray as from_numpy
from jax.numpy import ravel as flatten
from jax.scipy.integrate import trapezoid
from jax.scipy.integrate import trapezoid as trapz
from jax.scipy.special import erf, gamma, gammaln, polygamma

from .._backend_config import jax_atol as atol
from .._backend_config import jax_rtol as rtol
from . import fft  # For PyRecEst
from . import signal  # For PyRecEst
from . import spatial  # For PyRecEst
from . import autodiff, linalg, random
from ._dtype import as_dtype, set_default_dtype


def _asarray_sequence(seq):
    return tuple(_jnp.asarray(item) for item in seq)


def concatenate(seq, axis=0, dtype=None):
    return _jnp.concatenate(_asarray_sequence(seq), axis=axis, dtype=dtype)


def stack(seq, axis=0, dtype=None):
    return _jnp.stack(_asarray_sequence(seq), axis=axis, dtype=dtype)


def flip(m, axis=None):
    return _jnp.flip(_jnp.asarray(m), axis=axis)


def sort(a, axis=-1, **kwargs):
    return _jnp.sort(_jnp.asarray(a), axis=axis, **kwargs)


def unique(ar, *args, **kwargs):
    return _jnp.unique(_jnp.asarray(ar), *args, **kwargs)


def _asarray_or_none(value):
    return None if value is None else _jnp.asarray(value)


def cov(
    m,
    y=None,
    rowvar=True,
    bias=False,
    ddof=None,
    fweights=None,
    aweights=None,
    dtype=None,
):
    return _jnp.cov(
        _jnp.asarray(m),
        y=_asarray_or_none(y),
        rowvar=rowvar,
        bias=bias,
        ddof=ddof,
        fweights=_asarray_or_none(fweights),
        aweights=_asarray_or_none(aweights),
        dtype=dtype,
    )


def diagonal(a, offset=0, axis1=0, axis2=1):
    return _jnp.diagonal(_jnp.asarray(a), offset=offset, axis1=axis1, axis2=axis2)


def squeeze(a, axis=None):
    return _jnp.squeeze(_jnp.asarray(a), axis=axis)


def trace(a, offset=0, axis1=-2, axis2=-1, dtype=None, out=None):
    return _jnp.trace(
        _jnp.asarray(a),
        offset=offset,
        axis1=axis1,
        axis2=axis2,
        dtype=dtype,
        out=out,
    )


def tril(m, k=0):
    return _jnp.tril(_jnp.asarray(m), k=k)


def triu(m, k=0):
    return _jnp.triu(_jnp.asarray(m), k=k)


def argmax(a, axis=None, out=None, keepdims=False, **kwargs):
    result = _jnp.argmax(_jnp.asarray(a), axis=axis, keepdims=keepdims, **kwargs)
    if out is not None:
        return out.at[...].set(result)
    return result


def argmin(a, axis=None, out=None, keepdims=False, **kwargs):
    result = _jnp.argmin(_jnp.asarray(a), axis=axis, keepdims=keepdims, **kwargs)
    if out is not None:
        return out.at[...].set(result)
    return result


def convert_to_wider_dtype(*args, **kwargs):
    raise NotImplementedError(
        "The function convert_to_wider_dtype is not supported in this JAX backend."
    )


def get_default_dtype(*args, **kwargs):
    raise NotImplementedError(
        "The function get_default_dtype is not supported in this JAX backend."
    )


def get_default_cdtype(*args, **kwargs):
    raise NotImplementedError(
        "The function get_default_cdtype is not supported in this JAX backend."
    )


def to_ndarray(x, to_ndim, axis=0):
    """
    Convert an input to a JAX array and adjust its dimensionality if necessary.

    Parameters
    ----------
    x : array-like or scalar
        Input data, which could be a list, tuple, scalar, or an existing JAX array.
    to_ndim : int
        Target number of dimensions for the output array.
    axis : int, optional
        The axis along which a new dimension should be inserted, if needed.

    Returns
    -------
    x : jax.numpy.ndarray
        A JAX array with the desired number of dimensions.
    """
    # Ensure the input is a JAX array
    if not isinstance(x, _jnp.ndarray):
        x = _jnp.array(x)

    if x.ndim > to_ndim:
        raise ValueError("The ndim cannot be adapted properly.")

    while x.ndim < to_ndim:
        x = _jnp.expand_dims(x, axis=axis)

    return x


def take(
    a,
    indices,
    axis=None,
    out=None,
    mode=None,
    unique_indices=False,
    indices_are_sorted=False,
    fill_value=None,
):
    return _jnp.take(
        a,
        _jnp.asarray(indices),
        axis=axis,
        out=out,
        mode=mode,
        unique_indices=unique_indices,
        indices_are_sorted=indices_are_sorted,
        fill_value=fill_value,
    )


def _is_boolean_index(indices):
    if isinstance(indices, (bool, _jnp.bool_)):
        return True
    if isinstance(indices, (list, tuple)):
        return bool(indices) and _is_boolean_index(indices[0])
    if isinstance(indices, _jnp.ndarray):
        return indices.dtype in (_jnp.bool_, _jnp.uint8)
    return False


def _is_iterable_index(indices):
    return isinstance(indices, (list, tuple)) or (
        isinstance(indices, _jnp.ndarray) and indices.ndim > 0
    )


def _is_empty_index_sequence(indices):
    return _is_iterable_index(indices) and len(indices) == 0


def _assignment_index_length(indices, zip_indices):
    if zip_indices:
        return len(indices)
    if isinstance(indices, tuple) and all(
        isinstance(index, (_numbers.Integral, _jnp.integer)) for index in indices
    ):
        return 1
    return len(indices) if _is_iterable_index(indices) else 1


def _assignment_value_length(values):
    return len(values) if _is_iterable_index(values) else 1


def _apply_assignment(x_new, indices, values, *, accumulate):
    if accumulate:
        return x_new.at[indices].add(values)
    return x_new.at[indices].set(values)


def assignment(x, values, indices, axis=0):
    x_new = array(x)
    if _is_empty_index_sequence(indices):
        return copy(x_new)

    values = _jnp.asarray(values, dtype=x_new.dtype)
    use_vectorization = hasattr(indices, "__len__") and len(indices) < ndim(x_new)
    if _is_boolean_index(indices):
        return x_new.at[_jnp.asarray(indices, dtype=bool)].set(values)
    zip_indices = (
        _is_iterable_index(indices)
        and len(indices) > 0
        and _is_iterable_index(indices[0])
    )
    len_indices = _assignment_index_length(indices, zip_indices)
    if zip_indices:
        indices = tuple(zip(*indices))
    if not use_vectorization:
        len_values = _assignment_value_length(values)
        if len_values > 1 and len_values != len_indices:
            raise ValueError("Either one value or as many values as indices")
        x_new = _apply_assignment(x_new, indices, values, accumulate=False)
    else:
        indices = tuple(list(indices[:axis]) + [slice(None)] + list(indices[axis:]))
        x_new = x_new.at[indices].set(values)
    return x_new


def assignment_by_sum(x, values, indices, axis=0):
    x_new = array(x)
    if _is_empty_index_sequence(indices):
        return copy(x_new)

    values = _jnp.asarray(values, dtype=x_new.dtype)
    use_vectorization = hasattr(indices, "__len__") and len(indices) < ndim(x_new)
    if _is_boolean_index(indices):
        return x_new.at[_jnp.asarray(indices, dtype=bool)].add(values)
    zip_indices = (
        _is_iterable_index(indices)
        and len(indices) > 0
        and _is_iterable_index(indices[0])
    )
    len_indices = _assignment_index_length(indices, zip_indices)
    if zip_indices:
        indices = tuple(zip(*indices))
    if not use_vectorization:
        len_values = _assignment_value_length(values)
        if len_values > 1 and len_values != len_indices:
            raise ValueError("Either one value or as many values as indices")
        x_new = _apply_assignment(x_new, indices, values, accumulate=True)
    else:
        indices = tuple(list(indices[:axis]) + [slice(None)] + list(indices[axis:]))
        x_new = x_new.at[indices].add(values)
    return x_new


def is_array(x):
    return isinstance(x, _jnp.ndarray)


def is_bool(x):
    return is_array(x) and x.dtype == _jnp.bool_


def is_complex(x):
    return is_array(x) and _jnp.issubdtype(x.dtype, _jnp.complexfloating)


def is_floating(x):
    return is_array(x) and _jnp.issubdtype(x.dtype, _jnp.floating)
