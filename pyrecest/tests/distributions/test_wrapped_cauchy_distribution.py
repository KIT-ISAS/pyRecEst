import unittest
from math import pi

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import arange, array
from pyrecest.distributions.circle.custom_circular_distribution import (
    CustomCircularDistribution,
)
from pyrecest.distributions.circle.wrapped_cauchy_distribution import (
    WrappedCauchyDistribution,
)


class WrappedCauchyDistributionTest(unittest.TestCase):
    def setUp(self):
        self.mu = 0.0
        self.gamma = 0.5
        self.xs = arange(10)

    def test_pdf(self):
        dist = WrappedCauchyDistribution(self.mu, self.gamma)

        def pdf_wrapped(x, mu, gamma, terms=2000):
            summation = 0
            for k in range(-terms, terms + 1):
                summation += gamma / (pi * (gamma**2 + (x - mu + 2.0 * pi * k) ** 2))
            return summation

        custom_wrapped = CustomCircularDistribution(
            lambda xs: array([pdf_wrapped(x, self.mu, self.gamma) for x in xs])
        )

        npt.assert_allclose(
            dist.pdf(xs=self.xs), custom_wrapped.pdf(xs=self.xs), atol=0.0001
        )

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_cdf(self):
        dist = WrappedCauchyDistribution(self.mu, self.gamma)
        npt.assert_allclose(dist.cdf(array([1.0])), dist.integrate(array([0.0, 1.0])))


if __name__ == "__main__":
    unittest.main()
