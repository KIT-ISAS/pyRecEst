import unittest

import numpy as np
from pyrecest.distributions import WNDistribution


class WNDistributionTest(unittest.TestCase):
    def test_WNDistribution(self):
        mu = np.array(3)
        sigma = np.array(1.5)
        wn = WNDistribution(mu, sigma)

        # test pdf
        def approx_with_wrapping(x):
            total = 0
            for k in range(-20, 21):
                total += np.exp(-((x - mu + 2 * np.pi * k) ** 2) / (2 * sigma**2))
            return 1 / np.sqrt(2 * np.pi) / sigma * total

        self.assertAlmostEqual(wn.pdf(mu), approx_with_wrapping(mu), places=10)
        self.assertAlmostEqual(wn.pdf(mu - 1), approx_with_wrapping(mu - 1), places=10)
        self.assertAlmostEqual(wn.pdf(mu + 2), approx_with_wrapping(mu + 2), places=10)
        x = np.arange(0, 7)
        self.assertTrue(
            np.allclose(
                wn.pdf(x), np.array([approx_with_wrapping(xi) for xi in x]), rtol=1e-10
            )
        )

        # test pdf with large sigma
        wn_large_sigma = WNDistribution(0, 100)
        fx = np.ones_like(x) / (2 * np.pi)
        self.assertTrue(np.allclose(wn_large_sigma.pdf(x), fx, rtol=1e-10))


if __name__ == "__main__":
    unittest.main()
