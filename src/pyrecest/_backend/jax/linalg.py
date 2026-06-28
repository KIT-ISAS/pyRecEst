"""JAX-based linear algebra backend."""

import jax.numpy as _jnp
from jax.numpy.linalg import (  # NOQA
    cholesky,
    det,
    eig,
    eigh,
    eigvalsh,
    inv,
    matrix_power,
    matrix_rank,
    norm,
    pinv,
    solve,
    svd,
)
from jax.scipy.linalg import block_diag  # For PyRecEst
from jax.scipy.linalg import (
    expm,
    polar,
    sqrtm,
)

from .._backend_config import jax_atol as atol

unsupported_functions = [
    "fractional_matrix_power",
    "logm",
    "quadratic_assignment",
    "solve_sylvester",
]


def _unsupported_function(name):
    """Create an unsupported-function shim that preserves facade identity."""

    def _raise_unsupported(*args, **kwargs):
        del args, kwargs
        raise NotImplementedError(f"{name} is not supported in this JAX backend.")

    _raise_unsupported.__name__ = name
    _raise_unsupported.__qualname__ = name
    _raise_unsupported.__doc__ = f"Unsupported JAX-backend placeholder for ``{name}``."
    return _raise_unsupported


def qr(a, mode="reduced"):
    """Compute QR decomposition with NumPy-compatible mode handling."""

    a = _jnp.asarray(a)
    if mode == "economic":
        raw_result = _jnp.linalg.qr(a, mode="raw")
        return _jnp.swapaxes(raw_result[0], -2, -1)
    return _jnp.linalg.qr(a, mode=mode)


def is_single_matrix_pd(mat):
    """Check if a 2D square matrix is positive definite."""
    mat = _jnp.asarray(mat)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        return False

    if mat.dtype in (_jnp.complex64, _jnp.complex128):
        is_hermitian = _jnp.all(_jnp.abs(mat - _jnp.conj(_jnp.transpose(mat))) < atol)
        eigvals = _jnp.linalg.eigvalsh(mat)
        return _jnp.logical_and(is_hermitian, _jnp.min(_jnp.real(eigvals)) > 0)

    is_symmetric = _jnp.all(_jnp.abs(mat - _jnp.transpose(mat)) < atol)
    factor = _jnp.linalg.cholesky(mat)
    return _jnp.logical_and(is_symmetric, _jnp.all(_jnp.isfinite(factor)))


for func_name in unsupported_functions:
    globals()[func_name] = _unsupported_function(func_name)
