"""JAX-based linear algebra backend."""

import numpy as _np
import jax.numpy as _jnp
import jax.scipy.linalg as _jax_scipy_linalg

from .._backend_config import jax_atol as atol


def _as_linalg_array(value):
    """Convert PyRecEst array-like inputs before calling raw JAX linalg."""
    return _jnp.asarray(value)


def _normalize_norm_axis(axis):
    """Convert NumPy scalar-array axes to hashable values accepted by JAX."""
    if axis is None or isinstance(axis, (int, _np.integer)):
        return axis

    axis_array = _np.asarray(axis)
    if axis_array.ndim == 0:
        return int(axis_array.item())
    if axis_array.ndim == 1 and axis_array.size == 1:
        return int(axis_array.reshape(()).item())
    return axis


def _normalize_norm_arguments(args, kwargs):
    if len(args) >= 2:
        args = tuple(args)
        args = args[:1] + (_normalize_norm_axis(args[1]),) + args[2:]
    if "axis" in kwargs:
        kwargs = dict(kwargs)
        kwargs["axis"] = _normalize_norm_axis(kwargs["axis"])
    return args, kwargs


def cholesky(a, *args, **kwargs):
    return _jnp.linalg.cholesky(_as_linalg_array(a), *args, **kwargs)


def det(a, *args, **kwargs):
    return _jnp.linalg.det(_as_linalg_array(a), *args, **kwargs)


def eig(a, *args, **kwargs):
    return _jnp.linalg.eig(_as_linalg_array(a), *args, **kwargs)


def eigh(a, *args, **kwargs):
    return _jnp.linalg.eigh(_as_linalg_array(a), *args, **kwargs)


def eigvalsh(a, *args, **kwargs):
    return _jnp.linalg.eigvalsh(_as_linalg_array(a), *args, **kwargs)


def inv(a, *args, **kwargs):
    return _jnp.linalg.inv(_as_linalg_array(a), *args, **kwargs)


def matrix_power(a, n):
    return _jnp.linalg.matrix_power(_as_linalg_array(a), n)


def matrix_rank(a, *args, **kwargs):
    return _jnp.linalg.matrix_rank(_as_linalg_array(a), *args, **kwargs)


def norm(x, *args, **kwargs):
    args, kwargs = _normalize_norm_arguments(args, kwargs)
    return _jnp.linalg.norm(_as_linalg_array(x), *args, **kwargs)


def pinv(a, *args, **kwargs):
    return _jnp.linalg.pinv(_as_linalg_array(a), *args, **kwargs)


def qr(a, *args, **kwargs):
    return _jnp.linalg.qr(_as_linalg_array(a), *args, **kwargs)


def solve(a, b, *args, **kwargs):
    return _jnp.linalg.solve(_as_linalg_array(a), _as_linalg_array(b), *args, **kwargs)


def svd(a, *args, **kwargs):
    return _jnp.linalg.svd(_as_linalg_array(a), *args, **kwargs)


def block_diag(*arrs):
    return _jax_scipy_linalg.block_diag(*(_as_linalg_array(arr) for arr in arrs))


def expm(a, *args, **kwargs):
    return _jax_scipy_linalg.expm(_as_linalg_array(a), *args, **kwargs)


def polar(a, *args, **kwargs):
    return _jax_scipy_linalg.polar(_as_linalg_array(a), *args, **kwargs)


def sqrtm(a, *args, **kwargs):
    return _jax_scipy_linalg.sqrtm(_as_linalg_array(a), *args, **kwargs)


def solve_sylvester(a, b, q, *args, **kwargs):
    """Solve ``A X + X B = Q`` using JAX arrays and JAX SciPy."""
    return _jax_scipy_linalg.solve_sylvester(
        _as_linalg_array(a),
        _as_linalg_array(b),
        _as_linalg_array(q),
        *args,
        **kwargs,
    )


def _unsupported_function(name):
    """Create an unsupported-function shim that preserves facade identity."""

    def _raise_unsupported(*args, **kwargs):
        del args, kwargs
        raise NotImplementedError(f"{name} is not supported in this JAX backend.")

    _raise_unsupported.__name__ = name
    _raise_unsupported.__qualname__ = name
    _raise_unsupported.__doc__ = f"Unsupported JAX-backend placeholder for ``{name}``."
    return _raise_unsupported


fractional_matrix_power = _unsupported_function("fractional_matrix_power")
logm = _unsupported_function("logm")
quadratic_assignment = _unsupported_function("quadratic_assignment")


def is_single_matrix_pd(mat):
    """Check if a 2D square matrix is positive definite."""
    mat = _as_linalg_array(mat)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        return False

    if mat.dtype in (_jnp.complex64, _jnp.complex128):
        is_hermitian = _jnp.all(_jnp.abs(mat - _jnp.conj(_jnp.transpose(mat))) < atol)
        eigvals = _jnp.linalg.eigvalsh(mat)
        return _jnp.logical_and(is_hermitian, _jnp.min(_jnp.real(eigvals)) > 0)

    is_symmetric = _jnp.all(_jnp.abs(mat - _jnp.transpose(mat)) < atol)
    factor = _jnp.linalg.cholesky(mat)
    return _jnp.logical_and(is_symmetric, _jnp.all(_jnp.isfinite(factor)))
