import warnings

import numpy as np
import pytest

from pyrecest.distributions.conditional.sd_cond_sd_grid_distribution import (
    SdCondSdGridDistribution,
)
from pyrecest.distributions.conditional.sd_half_cond_sd_half_grid_distribution import (
    SdHalfCondSdHalfGridDistribution,
)
from pyrecest.distributions.conditional.td_cond_td_grid_distribution import (
    TdCondTdGridDistribution,
)


def _assert_columns_normalized(dist):
    integrals = dist._conditional_integrals()
    np.testing.assert_allclose(
        integrals, np.ones_like(integrals), rtol=1e-12, atol=1e-12
    )


def test_conditional_grid_normalize_returns_normalized_copy():
    grid = np.array([[0.0], [np.pi]])
    grid_values = np.array([[2.0, 6.0], [4.0, 10.0]])

    with pytest.warns(UserWarning, match="No normalisation is performed"):
        dist = TdCondTdGridDistribution(grid, grid_values)

    with pytest.warns(UserWarning, match="Normalizing each column"):
        normalized = dist.normalize()

    assert normalized is not dist
    np.testing.assert_allclose(dist.grid_values, grid_values)
    _assert_columns_normalized(normalized)


def test_conditional_grid_normalize_in_place_normalizes_columns():
    grid = np.array([[1.0, 0.0], [-1.0, 0.0]])
    grid_values = np.array([[1.0, 3.0], [2.0, 5.0]])

    with pytest.warns(UserWarning, match="No normalisation is performed"):
        dist = SdCondSdGridDistribution(grid, grid_values)

    with pytest.warns(UserWarning, match="Normalizing each column"):
        returned = dist.normalize_in_place()

    assert returned is dist
    _assert_columns_normalized(dist)


def test_conditional_grid_normalize_rejects_zero_integral_column():
    grid = np.array([[0.0], [np.pi]])
    grid_values = np.array([[0.0, 1.0], [0.0, 2.0]])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        dist = TdCondTdGridDistribution(grid, grid_values)

    with pytest.raises(ValueError, match="column integral is too close to zero"):
        dist.normalize()


def test_hemisphere_conditional_grid_normalize_uses_hemisphere_measure():
    grid = np.array([[1.0, 0.0], [-1.0, 0.0]])
    grid_values = np.array([[2.0, 3.0], [4.0, 5.0]])

    with pytest.warns(UserWarning):
        dist = SdHalfCondSdHalfGridDistribution(grid, grid_values)

    with pytest.warns(UserWarning, match="Normalizing each column"):
        normalized = dist.normalize()

    _assert_columns_normalized(normalized)
