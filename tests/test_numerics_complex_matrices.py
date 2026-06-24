import numpy as np
import pytest

from pyrecest.numerics import (
    assert_covariance_matrix,
    is_positive_semidefinite,
    is_symmetric,
    jittered_cholesky,
    nearest_symmetric_psd,
    symmetrize_matrix,
)


def test_covariance_helpers_reject_complex_matrices_without_losing_imaginary_part():
    matrix = np.array([[1.0 + 0.5j, 0.0], [0.0, 1.0]])

    assert not is_symmetric(matrix)
    assert not is_positive_semidefinite(matrix)

    with pytest.raises(ValueError, match="covariance must contain numeric values"):
        assert_covariance_matrix(matrix)
    with pytest.raises(ValueError, match="matrix must contain numeric values"):
        symmetrize_matrix(matrix)
    with pytest.raises(ValueError, match="matrix must contain numeric values"):
        nearest_symmetric_psd(matrix)
    with pytest.raises(ValueError, match="matrix must contain numeric values"):
        jittered_cholesky(matrix)
