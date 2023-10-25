import unittest
from math import pi

import matplotlib

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, linalg, sqrt
from pyrecest.distributions import (
    AbstractHypersphericalDistribution,
    VonMisesFisherDistribution,
)

matplotlib.use("Agg")


class AbstractHypersphericalDistributionTest(unittest.TestCase):
    def testIntegral2D(self):
        """Tests the integral calculation in 2D."""
        mu = array([1.0, 1.0, 2.0])
        mu = mu / linalg.norm(mu)
        kappa = 10.0
        vmf = VonMisesFisherDistribution(mu, kappa)
        self.assertAlmostEqual(vmf.integrate(), 1.0, delta=1e-8)

    def testIntegral3D(self):
        """Tests the integral calculation in 3D."""
        mu = array([1.0, 1.0, 2.0, 2.0])
        mu = mu / linalg.norm(mu)
        kappa = 10.0
        vmf = VonMisesFisherDistribution(mu, kappa)
        self.assertAlmostEqual(vmf.integrate(), 1.0, delta=1e-7)

    def testUnitSphereSurface(self):
        """Tests the unit sphere surface computation."""
        self.assertAlmostEqual(
            AbstractHypersphericalDistribution.compute_unit_hypersphere_surface(1),
            2.0 * pi,
            delta=1e-10,
        )
        self.assertAlmostEqual(
            AbstractHypersphericalDistribution.compute_unit_hypersphere_surface(2),
            4.0 * pi,
            delta=1e-10,
        )
        self.assertAlmostEqual(
            AbstractHypersphericalDistribution.compute_unit_hypersphere_surface(3),
            2.0 * pi**2,
            delta=1e-10,
        )

    def test_mean_direction_numerical(self):
        """Tests the numerical mean direction calculation."""
        mu = 1.0 / sqrt(2.0) * array([1.0, 1.0, 0.0])
        kappa = 10.0
        vmf = VonMisesFisherDistribution(mu, kappa)
        self.assertLess(linalg.norm(vmf.mean_direction_numerical() - mu), 1e-6)

    def test_plotting_error_free_2d(self):
        """Tests the plotting function"""

        mu = array([1.0, 1.0, 2.0])
        mu = mu / linalg.norm(mu)
        kappa = 10.0
        vmf = VonMisesFisherDistribution(mu, kappa)
        vmf.plot()


if __name__ == "__main__":
    unittest.main()
