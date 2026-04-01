import unittest

import numpy.testing as npt
from pyrecest.backend import array, pi
from pyrecest.distributions.circle.sine_skewed_distributions import (
    GSSVMDistribution,
    GeneralizedKSineSkewedVonMisesDistribution,
)


class TestGSSVMDistribution(unittest.TestCase):
    def test_initialization(self):
        """Test initialization with valid and invalid parameters."""
        dist = GSSVMDistribution(mu=pi, kappa=1.0, lambda_=0.5, n=1)
        self.assertAlmostEqual(float(dist.mu), float(pi))
        self.assertEqual(dist.kappa, 1.0)
        self.assertEqual(dist.lambda_, 0.5)
        self.assertEqual(dist.n, 1)
        self.assertEqual(dist.k, 1)

    def test_invalid_lambda(self):
        """lambda_ must be in [-1, 1]."""
        with self.assertRaises(AssertionError):
            GSSVMDistribution(mu=0.0, kappa=1.0, lambda_=1.5, n=1)

    def test_invalid_n_zero(self):
        """n must be a positive integer."""
        with self.assertRaises(AssertionError):
            GSSVMDistribution(mu=0.0, kappa=1.0, lambda_=0.5, n=0)

    def test_n_maps_to_m(self):
        """n property must equal the internal m parameter."""
        for n in (1, 2, 3, 4):
            dist = GSSVMDistribution(mu=0.0, kappa=1.0, lambda_=0.5, n=n)
            self.assertEqual(dist.n, n)
            self.assertEqual(dist.m, n)

    def test_pdf_non_negative(self):
        """PDF values must be non-negative everywhere."""
        xs = array([0.0, pi / 4, pi / 2, pi, 3 * pi / 2])
        for n in (1, 2, 3, 4):
            dist = GSSVMDistribution(mu=pi, kappa=1.0, lambda_=0.5, n=n)
            vals = dist.pdf(xs)
            self.assertTrue(all(vals >= 0), f"Negative PDF for n={n}")

    def test_pdf_matches_generalized_parent(self):
        """GSSVMDistribution PDF must match GeneralizedKSineSkewedVonMisesDistribution with k=1."""
        xs = array([0.0, pi / 4, pi / 2, pi, 3 * pi / 2])
        mu, kappa, lambda_ = 0.5, 2.0, 0.3
        for n in (1, 2, 3, 4):
            gssvm = GSSVMDistribution(mu=mu, kappa=kappa, lambda_=lambda_, n=n)
            parent = GeneralizedKSineSkewedVonMisesDistribution(
                mu=mu, kappa=kappa, lambda_=lambda_, k=1, m=n
            )
            npt.assert_array_almost_equal(gssvm.pdf(xs), parent.pdf(xs))

    def test_shift(self):
        """shift() must return a GSSVMDistribution with updated mu."""
        dist = GSSVMDistribution(mu=0.0, kappa=1.0, lambda_=0.5, n=2)
        shifted = dist.shift(array(pi / 2))
        self.assertIsInstance(shifted, GSSVMDistribution)
        self.assertAlmostEqual(float(shifted.mu), float(pi / 2))
        self.assertEqual(shifted.kappa, dist.kappa)
        self.assertEqual(shifted.lambda_, dist.lambda_)
        self.assertEqual(shifted.n, dist.n)

    def test_mu_wrapped(self):
        """mu should be wrapped to [0, 2*pi)."""
        dist = GSSVMDistribution(mu=3 * pi, kappa=1.0, lambda_=0.0, n=1)
        self.assertAlmostEqual(float(dist.mu), float(pi))

    def test_n_unsupported_raises(self):
        """n > 4 is not yet implemented."""
        dist = GSSVMDistribution(mu=0.0, kappa=1.0, lambda_=0.5, n=5)
        with self.assertRaises(NotImplementedError):
            dist.pdf(array(0.0))


if __name__ == "__main__":
    unittest.main()
