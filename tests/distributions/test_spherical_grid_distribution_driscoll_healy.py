import importlib.util
import unittest
import warnings

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import all, array, ones
from pyrecest.distributions.hypersphere_subset.spherical_grid_distribution import (
    SphericalGridDistribution,
)

pyshtools_installed = importlib.util.find_spec("pyshtools") is not None


class TestSphericalGridDistributionDriscollHealy(unittest.TestCase):
    @unittest.skipIf(not pyshtools_installed, "pyshtools is not installed")
    def test_from_function_driscoll_healy_unpacks_grid_and_uses_integer_degree(self):
        """Driscoll-Healy construction should produce a valid S² grid distribution."""

        def constant_density(xs):
            return ones(xs.shape[0])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            distribution = SphericalGridDistribution.from_function(
                constant_density,
                no_of_grid_points=91,
                grid_type="driscoll_healy",
            )

        self.assertEqual(distribution.grid_type, "driscoll_healy")
        self.assertEqual(distribution.grid.shape, (91, 3))
        self.assertEqual(distribution.grid_values.shape, (91,))
        self.assertTrue(all(distribution.grid_values == 1.0))
        npt.assert_allclose(distribution.pdf(array([0.0, 0.0, 1.0])), 1.0)


if __name__ == "__main__":
    unittest.main()
