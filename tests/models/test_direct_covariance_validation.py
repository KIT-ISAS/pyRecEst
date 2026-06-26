from __future__ import annotations

import numpy as np
import pytest
from pyrecest.models import MaskedLinearMeasurementModel, WeakDimensionMeasurementModel


def test_direct_measurement_covariances_reject_overflowed_values() -> None:
    with np.errstate(over="ignore"):
        overflowed = np.array([[np.finfo(float).max]]) * 2.0
    invalid_covariances = (overflowed, -overflowed)

    for covariance in invalid_covariances:
        with pytest.raises(ValueError, match="finite real numeric values"):
            MaskedLinearMeasurementModel(
                state_dim=1,
                observed_dims=[0],
                measurement_noise_cov=covariance,
            )
        with pytest.raises(ValueError, match="finite real numeric values"):
            WeakDimensionMeasurementModel(
                np.eye(1),
                measurement_noise_cov=covariance,
            )
