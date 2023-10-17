from math import pi
from pyrecest.backend import meshgrid
from pyrecest.backend import linspace
from pyrecest.backend import eye
from pyrecest.backend import array
import unittest

import numpy as np
from pyrecest.distributions import CustomLinearDistribution, GaussianDistribution
from pyrecest.distributions.nonperiodic.gaussian_mixture import GaussianMixture


class CustomLinearDistributionTest(unittest.TestCase):
    def setUp(self):
        g1 = GaussianDistribution(array([1, 1]), eye(2))
        g2 = GaussianDistribution(array([-3, -3]), eye(2))
        self.gm = GaussianMixture([g1, g2], array([0.7, 0.3]))

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
        x, y = meshgrid(linspace(0, 2 * pi, 10), linspace(0, 2 * pi, 10))
        np.testing.assert_allclose(
            dist1.pdf(np.column_stack((x.ravel(), y.ravel()))),
            dist2.pdf(np.column_stack((x.ravel(), y.ravel()))),
            atol=tol,
        )


if __name__ == "__main__":
    unittest.main()