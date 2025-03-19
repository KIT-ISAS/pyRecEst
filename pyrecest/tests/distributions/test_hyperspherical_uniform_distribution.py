"""Test for uniform distribution on the hypersphere"""

# pylint: disable=no-name-in-module,no-member
import unittest

import numpy.testing as npt
import pyrecest.backend
from pyrecest.backend import linalg, log, ones, random
from pyrecest.distributions import (
    AbstractHypersphericalDistribution,
    HypersphericalUniformDistribution,
)


class HypersphericalUniformDistributionTest(unittest.TestCase):
    @unittest.skipIf(
        pyrecest.backend.__name__ == "pyrecest.jax",
        "Test not supported for this backend",
    )
    def test_integrate_2d(self):
        hud = HypersphericalUniformDistribution(2)
        npt.assert_allclose(hud.integrate(), 1, atol=1e-6)

    @unittest.skipIf(
        pyrecest.backend.__name__ == "pyrecest.jax",
        "Test not supported for this backend",
    )
    def test_integrate_3d(self):
        hud = HypersphericalUniformDistribution(3)
        npt.assert_allclose(hud.integrate(), 1, atol=1e-6)

    def test_pdf(self):
        random.seed(0)
        for dim in range(2, 5):
            hud = HypersphericalUniformDistribution(dim)
            x = random.uniform(size=(dim + 1,))
            x = x / linalg.norm(x)
            npt.assert_allclose(
                hud.pdf(x),
                1
                / AbstractHypersphericalDistribution.compute_unit_hypersphere_surface(
                    dim
                ),
                atol=1e-10,
            )

    def test_sample(self):
        for dim in range(2, 5):
            hud = HypersphericalUniformDistribution(dim)
            n = 10
            samples = hud.sample(n)
            self.assertEqual(samples.shape, (n, hud.dim + 1))
            npt.assert_allclose(linalg.norm(samples, axis=1), ones(n), rtol=5e-7)

    def test_ln_pdf(self):
        """Test if ln_pdf returns the correct logarithm of the probability density."""
        hud = HypersphericalUniformDistribution(3)
        n = 10
        samples = hud.sample(n)
        # Assert that the computed values are close to the expected values
        npt.assert_array_almost_equal(
            hud.ln_pdf(samples),
            log(hud.pdf(samples)),
            decimal=10,
            err_msg="ln_pdf does not return correct log probabilities.",
        )


if __name__ == "__main__":
    unittest.main()
