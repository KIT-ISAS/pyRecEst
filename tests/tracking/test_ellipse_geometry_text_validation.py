import numpy as np
import pytest
from pyrecest.tracking import (
    canonicalize_ellipse_axes,
    project_symmetric_covariance,
    shape_from_extent_matrix,
)


@pytest.mark.parametrize("floor_value", ["0.0", np.array("0.0")])
def test_ellipse_numeric_floors_reject_text_scalars(floor_value) -> None:
    with pytest.raises(ValueError, match="minimum_eigenvalue"):
        project_symmetric_covariance(
            np.eye(2),
            minimum_eigenvalue=floor_value,
        )

    with pytest.raises(ValueError, match="minimum_axis_length"):
        shape_from_extent_matrix(
            np.eye(2),
            minimum_axis_length=floor_value,
        )

    with pytest.raises(ValueError, match="minimum_axis_length"):
        canonicalize_ellipse_axes(
            np.array([1.0, 2.0]),
            minimum_axis_length=floor_value,
        )
