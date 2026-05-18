import unittest

import numpy.testing as npt
import pyrecest
from pyrecest.backend import array, ones
from pyrecest.distributions.hypersphere_subset.hyperspherical_grid_distribution import (
    HypersphericalGridDistribution,
)


class HypersphereSubsetGridConstructorTest(unittest.TestCase):
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on this backend",
    )
    def test_constructor_normalizes_grid_values_in_place(self):
        grid = array(
            [
                [1.0, 0.0],
                [-1.0, 0.0],
            ]
        )
        grid_values = ones(2)

        dist = HypersphericalGridDistribution(grid, grid_values)

        npt.assert_allclose(dist.integrate(), 1.0, atol=1e-12, rtol=0.0)


if __name__ == "__main__":
    unittest.main()
