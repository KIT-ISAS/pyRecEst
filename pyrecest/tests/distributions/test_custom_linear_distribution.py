import unittest
import numpy as np
from pyrecest.distributions import CustomLinearDistribution
from pyrecest.distributions import GaussianDistribution
from pyrecest.distributions.nonperiodic.gaussian_mixture import GaussianMixture

class CustomLinearDistributionTest(unittest.TestCase):

    def setUp(self):
        g1 = GaussianDistribution(np.array([1, 1]), np.eye(2))
        g2 = GaussianDistribution(np.array([-3, -3]), np.eye(2))
        self.gm = GaussianMixture([g1, g2], [0.7, 0.3])

    def test_init_and_mean(self):
        cld = CustomLinearDistribution.from_distribution(self.gm)
        self.verify_pdf_equal(cld, self.gm, 1e-14)

    def test_integrate(self):
        cld = CustomLinearDistribution.from_distribution(self.gm)
        self.assertAlmostEqual(cld.integrate(), 1, delta=1e-10)

    def test_normalize(self):
        self.gm.w = self.gm.w / 2
        cld = CustomLinearDistribution.from_distribution(self.gm)
        self.assertAlmostEqual(cld.integrate(), 0.5, delta=1e-10)

    @staticmethod
    def verify_pdf_equal(dist1, dist2, tol):
        x, y = np.meshgrid(np.linspace(0, 2 * np.pi, 10), np.linspace(0, 2 * np.pi, 10))
        np.testing.assert_allclose(dist1.pdf(np.column_stack((x.ravel(), y.ravel()))), dist2.pdf(np.column_stack((x.ravel(), y.ravel()))), atol=tol)

if __name__ == '__main__':
    unittest.main()