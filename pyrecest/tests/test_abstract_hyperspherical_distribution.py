import unittest

import matplotlib
import numpy as np
from pyrecest.distributions import (
    AbstractHypersphericalDistribution,
    BinghamDistribution,
    VMFDistribution,
)
import matplotlib.pyplot as plt
matplotlib.use("Agg")



class AbstractHypersphericalDistributionTest(unittest.TestCase):
    def testIntegral2D(self):
        mu = np.array([1, 1, 2])
        mu = mu / np.linalg.norm(mu)
        kappa = 10
        vmf = VMFDistribution(mu, kappa)
        self.assertAlmostEqual(vmf.integrate(), 1, delta=1e-8)

    def testIntegral3D(self):
        mu = np.array([1, 1, 2, 2])
        mu = mu / np.linalg.norm(mu)
        kappa = 10
        vmf = VMFDistribution(mu, kappa)
        self.assertAlmostEqual(vmf.integrate(), 1, delta=1e-7)

    def testUnitSphereSurface(self):
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
        mu = 1 / np.sqrt(2) * np.array([1, 1, 0])
        kappa = 10
        vmf = VMFDistribution(mu, kappa)
        self.assertLess(np.linalg.norm(vmf.mean_direction_numerical() - mu), 1e-6)

    def test_mode_numerical_3D(self):
        M = np.eye(4)
        Z = np.array([-10, -2, -1, 0])
        bd = BinghamDistribution(Z, M)
        mom_num = bd.moment_numerical()
        mom_ana = bd.moment()
        self.assertLess(np.linalg.norm(mom_num - mom_ana), 0.003)

    def test_plotting_error_free_2d(self):
        mu = np.array([1, 1, 2])
        mu = mu / np.linalg.norm(mu)
        kappa = 10
        vmf = VMFDistribution(mu, kappa)
        vmf.plot()
        plt.close()


if __name__ == "__main__":
    unittest.main()
