""" Test for uniform distribution on the hypersphere """
import unittest

import numpy as np
from pyrecest.distributions import (
    AbstractHypersphericalDistribution,
    HypersphericalUniformDistribution,
)


class HypersphericalUniformDistributionTest(unittest.TestCase):
    def test_integrate_2d(self):
        hud = HypersphericalUniformDistribution(2)
        self.assertAlmostEqual(hud.integrate(), 1, delta=1e-6)

    def test_integrate_3d(self):
        hud = HypersphericalUniformDistribution(3)
        self.assertAlmostEqual(hud.integrate(), 1, delta=1e-6)

    def test_pdf(self):
        np.random.seed(0)
        for dim in range(2, 5):
            hud = HypersphericalUniformDistribution(dim)
            x = np.random.rand(dim + 1, 1)
            x = x / np.linalg.norm(x)
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
            self.assertTrue(
                np.allclose(np.linalg.norm(samples, axis=1), np.ones(n), rtol=1e-10)
            )


if __name__ == "__main__":
    unittest.main()
