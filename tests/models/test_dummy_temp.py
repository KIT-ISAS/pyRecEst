from __future__ import annotations

import numpy as np
import pytest
from pyrecest.models import MaskedLinearMeasurementModel, WeakDimensionMeasurementModel


def test_direct_measurement_covariances_reject_nonfinite_values() -> None:
    with np.errstate(divide="ignore", invalid="ignore"):
        invalid_covariances = (
            np.array([[1.0]]) / 0.0,
            np.array([[0.0]]) / 0.0,
        )

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
