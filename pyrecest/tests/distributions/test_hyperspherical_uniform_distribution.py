""" Test for uniform distribution on the hypersphere """
# pylint: disable=no-name-in-module,no-member
import unittest

from pyrecest.backend import allclose, linalg, ones, random
from pyrecest.distributions import (
    AbstractHypersphericalDistribution,
    HypersphericalUniformDistribution,
)
import numpy.testing as npt

class HypersphericalUniformDistributionTest(unittest.TestCase):
    def test_integrate_2d(self):
        hud = HypersphericalUniformDistribution(2)
        self.assertAlmostEqual(hud.integrate(), 1, delta=1e-6)

    def test_integrate_3d(self):
        hud = HypersphericalUniformDistribution(3)
        self.assertAlmostEqual(hud.integrate(), 1, delta=1e-6)

    def test_pdf(self):
        random.seed(0)
        for dim in range(2, 5):
            hud = HypersphericalUniformDistribution(dim)
            x = random.uniform(size=(dim + 1,))
            x = x / linalg.norm(x)
            self.assertAlmostEqual(
                hud.pdf(x),
                1
                / AbstractHypersphericalDistribution.compute_unit_hypersphere_surface(
                    dim
                ),
                delta=1e-10,
            )

    def test_sample(self):
        for dim in range(2, 5):
            hud = HypersphericalUniformDistribution(dim)
            n = 10
            samples = hud.sample(n)
            self.assertEqual(samples.shape, (n, hud.dim + 1))
            npt.assert_allclose(linalg.norm(samples, axis=1), ones(n), rtol=1e-7)


if __name__ == "__main__":
    unittest.main()
