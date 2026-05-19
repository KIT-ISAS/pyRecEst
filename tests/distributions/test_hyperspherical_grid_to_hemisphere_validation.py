import unittest

import pyrecest
from pyrecest.backend import array
from pyrecest.distributions.hypersphere_subset.hyperspherical_grid_distribution import (
    HypersphericalGridDistribution,
)


class HypersphericalGridToHemisphereValidationTest(unittest.TestCase):
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on this backend",
    )
    def test_invalid_even_grid_raises_valueerror_not_attributeerror(self):
        """Invalid even grids should fail with the documented validation error."""
        grid = array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ]
        )
        grid_values = array([1.0, 1.0, 1.0, 1.0])
        dist = HypersphericalGridDistribution(grid, grid_values)

        with self.assertRaises(ValueError) as cm:
            dist.to_hemisphere()

        self.assertIn("ToHemisphere:AsymmetricGrid", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
