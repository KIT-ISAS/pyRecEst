from pyrecest.backend import array
from pyrecest.backend import allclose
from pyrecest.backend import all
import unittest

import numpy as np
from pyrecest.distributions import VonMisesFisherDistribution
from pyrecest.distributions.hypersphere_subset.custom_hyperspherical_distribution import (
    CustomHypersphericalDistribution,
)


class CustomHypersphericalDistributionTest(unittest.TestCase):
    def setUp(self):
        self.vmf = VonMisesFisherDistribution(array([0, 0, 1]), 10)
        self.custom_hyperspherical_distribution = (
            CustomHypersphericalDistribution.from_distribution(self.vmf)
        )

    def test_simple_distribution(self):
        """Test that pdf function returns the correct size and values for given points."""
        p = self.custom_hyperspherical_distribution.pdf(np.asarray([1, 0, 0]))
        self.assertEqual(p.size, 1, "PDF size mismatch.")

        np.random.seed(10)
        points = np.random.randn(100, 3)
        points /= np.linalg.norm(points, axis=1, keepdims=True)

        self.assertTrue(
            allclose(
                self.custom_hyperspherical_distribution.pdf(points),
                self.vmf.pdf(points),
                atol=1e-5,
            ),
            "PDF values do not match.",
        )

    def test_integrate(self):
        """Test that the distribution integrates to 1."""
        self.assertAlmostEqual(
            self.custom_hyperspherical_distribution.integrate_numerically(),
            1,
            delta=1e-4,
            msg="Integration value mismatch.",
        )

    def test_from_distribution(self):
        """Test that the distribution can be created from another hyperspherical distribution."""
        dist = CustomHypersphericalDistribution.from_distribution(self.vmf)
        self.assertIsInstance(
            dist,
            CustomHypersphericalDistribution,
            "Type mismatch when creating from distribution.",
        )


if __name__ == "__main__":
    unittest.main()
