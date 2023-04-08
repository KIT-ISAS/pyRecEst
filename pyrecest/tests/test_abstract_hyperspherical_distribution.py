import unittest
import numpy as np
from pyrecest.distributions import VMFDistribution
from pyrecest.distributions import AbstractHypersphericalDistribution

class AbstractHypersphericalDistributionTest(unittest.TestCase):

    def testIntegral2D(self):
        mu = np.array([1, 1, 2])
        mu = mu/np.linalg.norm(mu)
        kappa = 10
        vmf = VMFDistribution(mu, kappa)
        self.assertAlmostEqual(vmf.integral(), 1, delta=1E-8)

    def testIntegral3D(self):
        mu = np.array([1, 1, 2, 2])
        mu = mu/np.linalg.norm(mu)
        kappa = 10
        vmf = VMFDistribution(mu, kappa)
        self.assertAlmostEqual(vmf.integral(), 1, delta=1E-7)
    """
    def testIntegral4D(self):
        mu = np.array([1, 1, 2, 2, 3])
        mu = mu/np.linalg.norm(mu)
        kappa = 10
        vmf = VMFDistribution(mu, kappa)
        self.assertAlmostEqual(vmf.integral(), 1, delta=1E-1)
    """
    def testUnitSphereSurface(self):
        self.assertAlmostEqual(
            AbstractHypersphericalDistribution.compute_unit_hypersphere_surface(1), 2*np.pi, delta=1E-10)
        self.assertAlmostEqual(
            AbstractHypersphericalDistribution.compute_unit_hypersphere_surface(2), 4*np.pi, delta=1E-10)
        self.assertAlmostEqual(AbstractHypersphericalDistribution.compute_unit_hypersphere_surface(3),
                               2*np.pi**2, delta=1E-10)
    """ Not yet implemented so no need to test
    def testMeanDirectionNumerical(self):
        mu = 1/np.sqrt(2)*np.array([1, 1, 0])
        kappa = 10
        vmf = VMFDistribution(mu, kappa)
        self.assertLess(np.linalg.norm(
            vmf.mean_direction_numerical() - mu), 1e-6)
    
    def testModeNumerical2D(self):
        M = np.eye(2)
        Z = np.array([-3, 0])
        bd = BinghamDistribution(Z, M)
        self.assertLess(np.linalg.norm(
            bd.moment_numerical() - bd.moment()), 0.001)

        phi = 0.7
        M = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
        Z = np.array([-5, 0])
        bd = BinghamDistribution(Z, M)
        self.assertLess(np.linalg.norm(
            bd.moment_numerical() - bd.moment()), 0.001)

    def testModeNumerical3D(self):
        M = np.eye(4)
        Z = np.array([-10, -2, -1, 0])
        bd = BinghamDistribution(Z, M)
        bd.F = bd.F*bd.integral_numerical()
        self.assertLess(np.linalg.norm(
            bd.moment_numerical() - bd.moment()), 0.003)
    """

if __name__ == '__main__':
    unittest.main()
