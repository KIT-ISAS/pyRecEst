import unittest

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pyrecest.distributions import (
    AbstractHypersphericalDistribution,
    VonMisesFisherDistribution,
)

matplotlib.use("Agg")


class AbstractHypersphericalDistributionTest(unittest.TestCase):
    def testIntegral2D(self):
        """Tests the integral calculation in 2D."""
        mu = np.array([1, 1, 2])
        mu = mu / np.linalg.norm(mu)
        kappa = 10
        vmf = VonMisesFisherDistribution(mu, kappa)
        self.assertAlmostEqual(vmf.integrate(), 1, delta=1e-8)

    def testIntegral3D(self):
        """Tests the integral calculation in 3D."""
        mu = np.array([1, 1, 2, 2])
        mu = mu / np.linalg.norm(mu)
        kappa = 10
        vmf = VonMisesFisherDistribution(mu, kappa)
        self.assertAlmostEqual(vmf.integrate(), 1, delta=1e-7)

    def testUnitSphereSurface(self):
        """Tests the unit sphere surface computation."""
        self.assertAlmostEqual(
            AbstractHypersphericalDistribution.compute_unit_hypersphere_surface(1),
            2 * np.pi,
            delta=1e-10,
        )
        self.assertAlmostEqual(
            AbstractHypersphericalDistribution.compute_unit_hypersphere_surface(2),
            4 * np.pi,
            delta=1e-10,
        )
        self.assertAlmostEqual(
            AbstractHypersphericalDistribution.compute_unit_hypersphere_surface(3),
            2 * np.pi**2,
            delta=1e-10,
        )

    def test_mean_direction_numerical(self):
        """Tests the numerical mean direction calculation."""
        mu = 1 / np.sqrt(2) * np.array([1, 1, 0])
        kappa = 10
        vmf = VonMisesFisherDistribution(mu, kappa)
        self.assertLess(np.linalg.norm(vmf.mean_direction_numerical() - mu), 1e-6)

    def test_plotting_error_free_2d(self):
        """Tests the numerical mode calculation."""
        
        mu = np.array([1, 1, 2])
        mu = mu / np.linalg.norm(mu)
        kappa = 10
        vmf = VonMisesFisherDistribution(mu, kappa)
        vmf.plot()
        plt.close()


if __name__ == "__main__":
    unittest.main()
