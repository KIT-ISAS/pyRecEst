import numpy as np
import pytest
from pyrecest.evaluation import (
    classify_inside_outside,
    surface_band_mask,
    surface_band_probability_from_signed_distance,
)


@pytest.mark.parametrize(
    "values",
    (
        [True, False],
        np.array([True, False]),
        ["0.0", "1.0"],
        np.array(["0.0", "1.0"]),
        np.array([1.0 + 1.0j, 2.0 + 0.0j]),
    ),
)
def test_surface_numeric_fields_reject_malformed_array_likes(values) -> None:
    with pytest.raises(ValueError, match="values"):
        surface_band_mask(values, 0.1)
    with pytest.raises(ValueError, match="values"):
        classify_inside_outside(values)
    with pytest.raises(ValueError, match="distance"):
        surface_band_probability_from_signed_distance(values, [0.1, 0.1], 0.1)
    with pytest.raises(ValueError, match="distance_std"):
        surface_band_probability_from_signed_distance([0.0, 0.1], values, 0.1)
