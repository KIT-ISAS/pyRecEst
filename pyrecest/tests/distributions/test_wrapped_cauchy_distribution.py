import unittest

import numpy as np
from pyrecest.distributions.circle.custom_circular_distribution import (
    CustomCircularDistribution,
)
from pyrecest.distributions.circle.wrapped_cauchy_distribution import (
    WrappedCauchyDistribution,
)


class WrappedCauchyDistributionTest(unittest.TestCase):
    def setUp(self):
        self.mu = 0
        self.gamma = 0.5
        self.xs = np.arange(10)

    def test_pdf(self):
        dist = WrappedCauchyDistribution(self.mu, self.gamma)

        def pdf_wrapped(x, mu, gamma, terms=2000):
            summation = 0
            for k in range(-terms, terms + 1):
                summation += gamma / (
                    np.pi * (gamma**2 + (x - mu + 2 * np.pi * k) ** 2)
                )
            return summation

        custom_wrapped = CustomCircularDistribution(
            lambda xs: np.array([pdf_wrapped(x, self.mu, self.gamma) for x in xs])
        )

        np.testing.assert_allclose(
            dist.pdf(xs=self.xs), custom_wrapped.pdf(xs=self.xs), atol=0.0001
        )

    def test_cdf(self):
        dist = WrappedCauchyDistribution(self.mu, self.gamma)
        np.testing.assert_allclose(
            dist.cdf(np.array([1])), dist.integrate(np.array([0, 1]))
        )


if __name__ == "__main__":
    unittest.main()
