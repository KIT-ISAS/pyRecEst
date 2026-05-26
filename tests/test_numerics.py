import numpy as np
import pytest
from pyrecest.exceptions import NumericalStabilityError
from pyrecest.numerics import (
    assert_covariance_matrix,
    is_positive_semidefinite,
    is_symmetric,
    jittered_cholesky,
    nearest_symmetric_psd,
    symmetrize_matrix,
)


def test_symmetrize_matrix_and_psd_projection():
    matrix = np.array([[1.0, 2.0], [0.0, -0.1]])
    symmetric = np.asarray(symmetrize_matrix(matrix))
    assert np.allclose(symmetric, symmetric.T)

    repaired = np.asarray(nearest_symmetric_psd(matrix))
    assert is_symmetric(repaired)
    assert is_positive_semidefinite(repaired)


def test_nearest_symmetric_psd_rejects_invalid_min_eigenvalue():
    matrix = np.eye(2)

    for min_eigenvalue in (-1.0, np.nan, np.inf):
        with pytest.raises(ValueError, match="min_eigenvalue"):
            nearest_symmetric_psd(matrix, min_eigenvalue=min_eigenvalue)


def test_jittered_cholesky_reports_jitter():
    matrix = np.array([[1.0, 0.0], [0.0, 0.0]])
    factor, jitter = jittered_cholesky(matrix)
    assert np.asarray(factor).shape == (2, 2)
    assert jitter > 0.0


def test_jittered_cholesky_rejects_invalid_retry_controls():
    matrix = np.eye(2)

    for initial_jitter in (0.0, -1e-12, np.nan, np.inf):
        with pytest.raises(ValueError, match="initial_jitter"):
            jittered_cholesky(matrix, initial_jitter=initial_jitter)

    for max_attempts in (-1, 1.5):
        with pytest.raises(ValueError, match="max_attempts"):
            jittered_cholesky(matrix, max_attempts=max_attempts)


def test_assert_covariance_matrix_rejects_non_psd():
    with pytest.raises(NumericalStabilityError):
        assert_covariance_matrix(np.array([[1.0, 0.0], [0.0, -1.0]]))
