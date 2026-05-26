import numpy as np
import pytest
from pyrecest.numerics import (
    assert_covariance_matrix,
    is_positive_semidefinite,
    is_symmetric,
)


@pytest.mark.parametrize("atol", [-1.0, np.nan, np.inf])
@pytest.mark.parametrize("validator", [is_symmetric, is_positive_semidefinite])
def test_matrix_predicates_reject_invalid_atol(atol, validator):
    with pytest.raises(ValueError, match="atol must be finite and non-negative"):
        validator(np.eye(2), atol=atol)


@pytest.mark.parametrize("atol", [-1.0, np.nan, np.inf])
def test_assert_covariance_matrix_rejects_invalid_atol(atol):
    with pytest.raises(ValueError, match="atol must be finite and non-negative"):
        assert_covariance_matrix(np.eye(2), atol=atol)
