import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, pi
from pyrecest.distributions.hypertorus.custom_hypertoroidal_distribution import (
    CustomHypertoroidalDistribution,
)


class AbstractCustomDistributionTest(unittest.TestCase):
    def test_normalize_verify_accepts_scalar_integral(self):
        dist = CustomHypertoroidalDistribution(
            lambda xs: 2.0 + 0.0 * xs[0],
            dim=1,
        )

        normalized = dist.normalize(verify=True)

        npt.assert_allclose(normalized.integrate(), 1.0, atol=1e-10)
        npt.assert_allclose(normalized.pdf(array([0.0])), 1.0 / (2.0 * pi))


if __name__ == "__main__":
    unittest.main()
