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


def test_jittered_cholesky_reports_jitter():
    matrix = np.array([[1.0, 0.0], [0.0, 0.0]])
    factor, jitter = jittered_cholesky(matrix)
    assert np.asarray(factor).shape == (2, 2)
    assert jitter > 0.0


def test_assert_covariance_matrix_rejects_non_psd():
    with pytest.raises(NumericalStabilityError):
        assert_covariance_matrix(np.array([[1.0, 0.0], [0.0, -1.0]]))


@pytest.mark.parametrize("atol", [-1.0, float("nan"), float("inf")])
@pytest.mark.parametrize("validator", [is_symmetric, is_positive_semidefinite])
def test_matrix_predicates_reject_invalid_atol(atol, validator):
    with pytest.raises(ValueError, match="atol must be finite and non-negative"):
        validator(np.eye(2), atol=atol)


@pytest.mark.parametrize("atol", [-1.0, float("nan"), float("inf")])
def test_assert_covariance_matrix_rejects_invalid_atol(atol):
    with pytest.raises(ValueError, match="atol must be finite and non-negative"):
        assert_covariance_matrix(np.eye(2), atol=atol)
