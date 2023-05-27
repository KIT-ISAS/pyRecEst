import unittest

import numpy as np
from pyrecest.distributions import WrappedNormalDistribution


class WrappedNormalDistributionTest(unittest.TestCase):

    def setUp(self):
        self.mu = np.array(3)
        self.sigma = np.array(1.5)
        self.wn = WrappedNormalDistribution(self.mu, self.sigma)

    def test_pdf_values_are_as_expected(self):
        """
        Test that the probability density function (pdf) returns expected values.
        """

        def approx_with_wrapping(x):
            k = np.arange(-20, 21)
            total = np.sum(np.exp(-((x - self.mu + 2 * np.pi * k) ** 2) / (2 * self.sigma**2)))
            return 1 / np.sqrt(2 * np.pi) / self.sigma * total

        test_points = [self.mu, self.mu - 1, self.mu + 2]
        for point in test_points:
            with self.subTest(x=point):
                self.assertAlmostEqual(self.wn.pdf(point), approx_with_wrapping(point), places=10)
        
        x = np.arange(0, 7)
        self.assertTrue(
            np.allclose(
                self.wn.pdf(x), np.array([approx_with_wrapping(xi) for xi in x]), rtol=1e-10
            )
        )

    def test_pdf_with_large_sigma_is_uniform(self):
        """
        Test that the pdf with large sigma is approximately a uniform distribution.
        """
        wn_large_sigma = WrappedNormalDistribution(0, 100)
        x = np.arange(0, 7)
        fx = np.ones_like(x) / (2 * np.pi)
        self.assertTrue(np.allclose(wn_large_sigma.pdf(x), fx, rtol=1e-10))


if __name__ == "__main__":
    unittest.main()