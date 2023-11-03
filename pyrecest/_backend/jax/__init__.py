"""Jax-based computation backend.
based on implementation by Emile Mathieu
for Riemannian Score-based SDE
"""
import jax.numpy as _jnp
from jax.numpy import (
    all,
    allclose,
    amax,
    amin,
    any,
    argmax,
    argmin,
    broadcast_arrays,
    broadcast_to,
    clip,
    complex64,
    complex128,
    concatenate,
    conj,
    cross,
    cumprod,
    cumsum,
    diag_indices,
    diagonal,
    einsum,
    empty_like,
    equal,
    expand_dims,
    flip,
    float32,
    float64,
    greater,
    hsplit,
    hstack,
    int32,
    int64,
    isclose,
    isnan,
    kron,
    less,
    less_equal,
    logical_and,
    logical_or,
    maximum,
    mean,
    meshgrid,
    minimum,
    moveaxis,
    ones_like,
    pad,
    prod,
    quantile,
    repeat,
    reshape,
    searchsorted,
    shape,
    sort,
    split,
    stack,
    std,
    sum,
    take,
    tile,
    transpose,
    trapz,
    tril,
    tril_indices,
    triu,
    triu_indices,
    uint8,
    unique,
    vstack,
    where,
    zeros_like,
    diag,
    diff,
    apply_along_axis,
    nonzero,
    column_stack,
    conj,
    atleast_1d,
    atleast_2d,
    dstack,
    full,
    isreal,
    triu,
    kron,
    angle,
    arctan,
    cov,
    count_nonzero,
    full_like,
    isinf,
    deg2rad,
    argsort,
    max,
    min,
    roll,
    dstack,
    abs,
    arange,
    abs,
    angle,
    arange,
    arccos,
    arccosh,
    arcsin,
    arctan2,
    arctanh,
    ceil,
    copy,
    cos,
    cosh,
    divide,
    dot,
    exp,
    floor,
    imag,
    log,
    matmul,
    mod,
    ndim,
    outer,
    power,
    real,
    sign,
    sin,
    sinh,
    sqrt,
    squeeze,
    tan,
    tanh,
    trace,
    vectorize,
    empty,
    eye,
    zeros,
    linspace,
    ones,
)
from jax import vmap

from jax import device_get as to_numpy

from jax.scipy.special import erf, gamma, polygamma

from jax.numpy import ravel as flatten
from jax.numpy import asarray as from_numpy

from .._backend_config import jax_atol as atol
from .._backend_config import jax_rtol as rtol


from . import autodiff
from . import linalg
from . import random
from . import fft

from jax.numpy import array

unsupported_functions = [
    'array_from_sparse',
    'assignment',
    'assignment_by_sum',
    'cast',
    'convert_to_wider_dtype',
    'get_default_dtype',
    'get_default_cdtype',
    'get_slice',
    'is_array',
    'is_complex',
    'ravel_tril_indices',
    'set_default_dtype',
    'to_ndarray',
]
for func_name in unsupported_functions:
    exec(f"{func_name} = lambda *args, **kwargs: NotImplementedError('This function is not supported in this JAX backend.')")




def as_dtype(array, dtype):
    """Change the data type of a given array.
    
    Parameters:
    - array: The array whose data type needs to be changed
    - dtype: The new data type
    
    Returns:
    A new array with the specified data type.
    """
    return _jnp.asarray(array, dtype=dtype)


# Check if dtype is floating-point
def is_floating(array):
    return _jnp.issubdtype(array.dtype, _jnp.floating)


# Check if dtype is boolean
def is_bool(array):
    return _jnp.issubdtype(array.dtype, _jnp.bool_)


# Matrix-vector multiplication
def matvec(matrix, vector):
    return _jnp.dot(matrix, vector)


# One-hot encoding
def one_hot(indices, depth):
    return _jnp.eye(depth)[indices]


# Scatter-add operation
def scatter_add(array, indices, updates):
    return _jnp.zeros_like(array).at[indices].add(updates)


# Set diagonal elements of a matrix
def set_diag(matrix, values):
    return matrix.at[_jnp.diag_indices_from(matrix)].set(values)


# Get lower triangle and flatten to vector
def tril_to_vec(matrix):
    return _jnp.tril(matrix).ravel()


# Get upper triangle and flatten to vector
def triu_to_vec(matrix):
    return _jnp.triu(matrix).ravel()


# Create diagonal matrix from vector
def vec_to_diag(vector):
    return _jnp.diag(vector)


# Create matrix from diagonal, upper triangular, and lower triangular parts
def mat_from_diag_triu_tril(diag, triu, tril):
    matrix = _jnp.diag(diag)
    matrix = matrix.at[_jnp.triu_indices_from(matrix, k=1)].set(triu)
    matrix = matrix.at[_jnp.tril_indices_from(matrix, k=-1)].set(tril)
    return matrix