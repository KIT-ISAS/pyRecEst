"""Numerical-contract helpers for covariance-like matrices."""

from __future__ import annotations

import numpy as np
from pyrecest.exceptions import (
    DimensionMismatchError,
    NumericalStabilityError,
    ShapeError,
)


def _to_numpy_array(value) -> np.ndarray:
    try:
        import pyrecest.backend as backend

        return np.asarray(backend.to_numpy(value), dtype=float)
    except (
        Exception
    ):  # pragma: no cover - fallback for source-tree bootstrap or unusual array objects
        return np.asarray(value, dtype=float)


def _from_numpy_array(value: np.ndarray):
    try:
        import pyrecest.backend as backend

        return backend.array(value)
    except Exception:  # pragma: no cover
        return value


def symmetrize_matrix(matrix):
    """Return ``0.5 * (matrix + matrix.T)`` in the active backend representation."""
    arr = _to_numpy_array(matrix)
    if arr.ndim != 2:
        raise ShapeError(f"Expected a matrix, got shape {arr.shape}.")
    return _from_numpy_array(0.5 * (arr + arr.T))


def is_symmetric(matrix, *, atol: float = 1e-10) -> bool:
    """Return whether a matrix is symmetric within an absolute tolerance."""
    arr = _to_numpy_array(matrix)
    return bool(
        arr.ndim == 2
        and arr.shape[0] == arr.shape[1]
        and np.allclose(arr, arr.T, atol=atol, rtol=0.0)
    )


def is_positive_semidefinite(matrix, *, atol: float = 1e-10) -> bool:
    """Return whether a symmetric matrix is positive semidefinite within tolerance."""
    arr = _to_numpy_array(matrix)
    if (
        arr.ndim != 2
        or arr.shape[0] != arr.shape[1]
        or not is_symmetric(arr, atol=atol)
    ):
        return False
    return bool(np.min(np.linalg.eigvalsh(arr)) >= -atol)


def nearest_symmetric_psd(matrix, *, min_eigenvalue: float = 0.0):
    """Project a symmetric matrix to the nearest eigenvalue-clipped PSD matrix.

    This helper is intended for diagnostics and controlled numerical repair. It
    should not silently replace validation in algorithms where invalid covariance
    matrices indicate a modeling error.
    """
    arr = _to_numpy_array(matrix)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ShapeError(f"Expected a square matrix, got shape {arr.shape}.")
    sym = 0.5 * (arr + arr.T)
    eigvals, eigvecs = np.linalg.eigh(sym)
    clipped = np.maximum(eigvals, min_eigenvalue)
    repaired = (eigvecs * clipped) @ eigvecs.T
    return _from_numpy_array(0.5 * (repaired + repaired.T))


def jittered_cholesky(matrix, *, initial_jitter: float = 1e-12, max_attempts: int = 8):
    """Return a Cholesky factor and the jitter used to obtain it.

    The function tries the raw matrix first, then repeatedly adds diagonal
    jitter. It raises :class:`NumericalStabilityError` if no factorization is
    found within ``max_attempts``.
    """
    arr = _to_numpy_array(matrix)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ShapeError(f"Expected a square matrix, got shape {arr.shape}.")
    sym = 0.5 * (arr + arr.T)
    eye = np.eye(sym.shape[0])
    jitter = 0.0
    for attempt in range(max_attempts + 1):
        try:
            factor = np.linalg.cholesky(sym + jitter * eye)
            return _from_numpy_array(factor), jitter
        except np.linalg.LinAlgError:
            jitter = initial_jitter if attempt == 0 else jitter * 10.0
    raise NumericalStabilityError(
        f"Cholesky factorization failed after {max_attempts} jitter attempts."
    )


def assert_covariance_matrix(
    matrix, *, name: str = "covariance", dim: int | None = None, atol: float = 1e-10
):
    """Validate a covariance matrix and return it in the active backend representation."""
    arr = _to_numpy_array(matrix)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ShapeError(f"{name} must be a square matrix, got shape {arr.shape}.")
    if dim is not None and arr.shape[0] != dim:
        raise DimensionMismatchError(
            f"{name} dimension {arr.shape[0]} does not match expected dimension {dim}."
        )
    if not is_symmetric(arr, atol=atol):
        raise NumericalStabilityError(f"{name} must be symmetric within atol={atol}.")
    if not is_positive_semidefinite(arr, atol=atol):
        raise NumericalStabilityError(
            f"{name} must be positive semidefinite within atol={atol}."
        )
    return _from_numpy_array(arr)


__all__ = [
    "assert_covariance_matrix",
    "is_positive_semidefinite",
    "is_symmetric",
    "jittered_cholesky",
    "nearest_symmetric_psd",
    "symmetrize_matrix",
]
