import math
import unittest

import pyrecest.backend

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, isfinite, pi
from pyrecest.distributions.hypersphere_subset.custom_hyperspherical_distribution import (
    CustomHypersphericalDistribution,
)
from pyrecest.distributions.hypertorus.custom_hypertoroidal_distribution import (
    CustomHypertoroidalDistribution,
)


@unittest.skipIf(
    pyrecest.backend.__backend_name__ != "numpy",
    reason="Numerical entropy integration is currently supported for numpy backend only.",
)
class ZeroDensityEntropyTest(unittest.TestCase):
    def test_hypertoroidal_entropy_handles_zero_density(self):
        def pdf(xs):
            xs = array(xs)
            return (xs[0] < pi) / pi

        dist = CustomHypertoroidalDistribution(pdf, 1)
        entropy = dist.entropy_numerical()

        self.assertTrue(isfinite(entropy))
        self.assertAlmostEqual(entropy, math.log(math.pi), delta=1e-8)

    def test_hyperspherical_entropy_handles_zero_density(self):
        def pdf(xs):
            xs = array(xs)
            return (xs[-1] >= 0.0) / (2.0 * pi)

        dist = CustomHypersphericalDistribution(pdf, 2)
        entropy = dist.entropy_numerical()

        self.assertTrue(isfinite(entropy))
        self.assertAlmostEqual(entropy, math.log(2.0 * math.pi), delta=1e-6)


if __name__ == "__main__":
    unittest.main()
