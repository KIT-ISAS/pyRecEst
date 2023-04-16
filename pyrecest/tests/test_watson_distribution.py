import unittest

import numpy as np
from pyrecest.distributions import WatsonDistribution


class TestWatsonDistribution(unittest.TestCase):
    def test_constructor(self):
        mu = np.array([1, 2, 3])
        mu = mu / np.linalg.norm(mu)
        kappa = 2
        w = WatsonDistribution(mu, kappa)

        self.assertIsInstance(w, WatsonDistribution)
        np.testing.assert_array_equal(w.mu, mu)
        self.assertEqual(w.kappa, kappa)
        self.assertEqual(w.dim, len(mu) - 1)

    def test_pdf(self):
        mu = np.array([1, 2, 3])
        mu = mu / np.linalg.norm(mu)
        kappa = 2
        w = WatsonDistribution(mu, kappa)

        xs = np.array(
            [[1, 0, 0], [1, 2, 2], [0, 1, 0], [0, 0, 1], [1, 1, 1], [-1, -1, -1]],
            dtype=float,
        )
        xs = xs / np.linalg.norm(xs, axis=1, keepdims=True)

        expected_pdf_values = np.array(
            [
                0.0388240901641662,
                0.229710245437696,
                0.0595974246790006,
                0.121741272709942,
                0.186880524436683,
                0.186880524436683,
            ]
        )

        pdf_values = w.pdf(xs)
        np.testing.assert_almost_equal(pdf_values, expected_pdf_values, decimal=5)

    def test_integral(self):
        mu = np.array([1, 2, 3])
        mu = mu / np.linalg.norm(mu)
        kappa = 2
        w = WatsonDistribution(mu, kappa)

        # Test integral
        self.assertAlmostEqual(w.integral(), 1, delta=1e-5)


if __name__ == "__main__":
    unittest.main()
