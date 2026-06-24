"""Regression tests for implicit-surface helpers with plain array-like inputs."""

import numpy as np
import pytest
from pyrecest.evaluation import (
    classify_inside_outside,
    surface_band_mask,
    surface_band_probability_from_signed_distance,
)


def test_surface_band_probability_accepts_plain_array_like_inputs() -> None:
    probability = surface_band_probability_from_signed_distance(
        [0.0, 0.2],
        [0.05, 0.05],
        0.1,
    )

    assert probability.shape == (2,)
    assert np.all(np.isfinite(probability))
    assert probability[0] > 0.9
    assert 0.0 <= probability[1] < probability[0]


@pytest.mark.parametrize(
    "bad_values",
    (
        [True, False],
        np.array([True, False]),
        ["0.0", "1.0"],
        np.array(["0.0", "1.0"]),
    ),
)
def test_surface_numeric_fields_reject_bool_and_text_array_likes(bad_values) -> None:
    with pytest.raises(ValueError, match="values"):
        surface_band_mask(bad_values, 0.1)
    with pytest.raises(ValueError, match="values"):
        classify_inside_outside(bad_values)
    with pytest.raises(ValueError, match="distance"):
        surface_band_probability_from_signed_distance(bad_values, [0.1, 0.1], 0.1)
    with pytest.raises(ValueError, match="distance_std"):
        surface_band_probability_from_signed_distance([0.0, 0.1], bad_values, 0.1)
