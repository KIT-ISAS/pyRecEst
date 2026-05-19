import unittest
from unittest.mock import patch

import numpy.testing as npt
from pyrecest.backend import array, asarray, ones, pi
from pyrecest.distributions.conditional.sd_half_cond_sd_half_grid_distribution import (
    SdHalfCondSdHalfGridDistribution,
)


def _mock_hyperhemisphere_grid(_grid_type, _no_of_grid_points, _manifold_dim):
    """Return a valid upper S1-half grid whose size differs from the request."""
    return (
        array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [-1.0, 0.0],
            ]
        ),
        {"scheme": "mock"},
    )


class SdHalfCondSdHalfGridDistributionFromFunctionTest(unittest.TestCase):
    def test_from_function_uses_returned_grid_size(self):
        """from_function must use grid.shape[0], not the density parameter."""
        uniform_value = 1.0 / pi

        def uniform_fun(a, _b):
            return ones(a.shape[0]) * uniform_value

        with patch(
            "pyrecest.sampling.hyperspherical_sampler.get_grid_hyperhemisphere",
            side_effect=_mock_hyperhemisphere_grid,
        ):
            dist = SdHalfCondSdHalfGridDistribution.from_function(
                uniform_fun,
                no_of_grid_points=2,
                fun_does_cartesian_product=False,
                grid_type="mock",
                dim=4,
            )

        self.assertEqual(asarray(dist.grid).shape, (3, 2))
        self.assertEqual(asarray(dist.grid_values).shape, (3, 3))
        npt.assert_allclose(asarray(dist.grid_values), asarray(ones((3, 3)) / pi))
