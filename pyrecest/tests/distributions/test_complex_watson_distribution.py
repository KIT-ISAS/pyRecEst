import math
import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member,redefined-builtin
from pyrecest.backend import abs, all, array, einsum, exp, linalg, ones, random, stack

from pyrecest.distributions.hypersphere_subset.complex_watson_distribution import (
    ComplexWatsonDistribution,
)


def _random_unit_vector(D):
    """Return a random unit vector in C^D."""
    z = random.normal(size=(D,)) + 1j * random.normal(size=(D,))
    return z / linalg.norm(z)


class TestComplexWatsonLogNorm(unittest.TestCase):
    """Tests for the log normalisation constant."""

    def test_scalar_input(self):
        val = ComplexWatsonDistribution.log_norm(3, 5.0)
        self.assertIsInstance(val, float)

    def test_array_input(self):
        kappas = array([0.1, 1.0, 10.0, 200.0])
        vals = ComplexWatsonDistribution.log_norm(3, kappas)
        self.assertEqual(vals.shape, (4,))

    def test_D2_low_kappa(self):
        # For low kappa, log C ~ log(2) + 2*log(pi) - log(Gamma(2)) + log(1) = log(2 pi^2)
        # so log_c = -log(2*pi^2)
        log_c = ComplexWatsonDistribution.log_norm(2, 1e-10)
        expected = -math.log(2 * math.pi**2)
        self.assertAlmostEqual(log_c, expected, places=5)

    def test_continuity_across_regimes(self):
        # Values should be continuous at regime boundaries 1/D and 100
        D = 3
        eps = 1e-4
        # Boundary kappa ~ 1/D = 1/3
        v1 = ComplexWatsonDistribution.log_norm(D, 1.0 / D - eps)
        v2 = ComplexWatsonDistribution.log_norm(D, 1.0 / D + eps)
        self.assertAlmostEqual(v1, v2, places=2)
        # Boundary kappa ~ 100
        v3 = ComplexWatsonDistribution.log_norm(D, 100.0 - eps)
        v4 = ComplexWatsonDistribution.log_norm(D, 100.0 + eps)
        self.assertAlmostEqual(v3, v4, places=2)


class TestComplexWatsonDistribution(unittest.TestCase):
    """Tests for ComplexWatsonDistribution."""

    def setUp(self):
        random.seed(0)

    def _unit_mu(self, D):
        z = random.normal(size=(D,)) + 1j * random.normal(size=(D,))
        return z / linalg.norm(z)

    def test_constructor(self):
        D = 3
        mu = self._unit_mu(D)
        dist = ComplexWatsonDistribution(mu, 2.0)
        self.assertEqual(dist.dim, D)
        self.assertAlmostEqual(dist.kappa, 2.0)
        npt.assert_array_almost_equal(dist.mu, mu)

    def test_pdf_positive(self):
        D = 3
        mu = self._unit_mu(D)
        dist = ComplexWatsonDistribution(mu, 5.0)
        Z = stack([_random_unit_vector(D) for _ in range(10)], axis=1)
        p = dist.pdf(Z)
        self.assertTrue(all(p > 0))

    def test_pdf_mode_is_maximum(self):
        """The PDF at the mode (mu) should be >= the PDF at any other point."""
        D = 3
        mu = self._unit_mu(D)
        dist = ComplexWatsonDistribution(mu, 10.0)
        Z = stack([_random_unit_vector(D) for _ in range(50)], axis=1)
        p_mode = dist.pdf(mu.reshape(-1, 1))[0]
        self.assertTrue(all(p_mode >= dist.pdf(Z) - 1e-10))

    def test_pdf_antipodal_symmetry(self):
        """The PDF should be the same at z and -z (|mu^H z|^2 = |mu^H (-z)|^2)."""
        D = 3
        mu = self._unit_mu(D)
        dist = ComplexWatsonDistribution(mu, 5.0)
        z = _random_unit_vector(D).reshape(-1, 1)
        npt.assert_allclose(dist.pdf(z), dist.pdf(-z), rtol=1e-10)

    def test_pdf_phase_invariance(self):
        """Multiplying z by a complex phase should not change the PDF."""
        D = 3
        mu = self._unit_mu(D)
        dist = ComplexWatsonDistribution(mu, 5.0)
        z = _random_unit_vector(D).reshape(-1, 1)
        phase = exp(1j * 1.23)
        npt.assert_allclose(dist.pdf(z), dist.pdf(phase * z), rtol=1e-10)

    def test_hypergeometric_ratio_bounds(self):
        D = 4
        r0 = ComplexWatsonDistribution.hypergeometric_ratio(0.0, D)
        r_large = ComplexWatsonDistribution.hypergeometric_ratio(1000.0, D)
        self.assertAlmostEqual(r0, 1.0 / D, places=5)
        self.assertGreater(r_large, 0.99)
        self.assertLessEqual(r_large, 1.0)

    def test_hypergeometric_ratio_inverse(self):
        D = 3
        for kappa in [0.5, 5.0, 50.0]:
            r = ComplexWatsonDistribution.hypergeometric_ratio(kappa, D)
            kappa_hat = ComplexWatsonDistribution.hypergeometric_ratio_inverse(r, D)
            self.assertAlmostEqual(kappa_hat, kappa, places=3)

    def test_sample_on_unit_sphere(self):
        D = 3
        mu = self._unit_mu(D)
        dist = ComplexWatsonDistribution(mu, 5.0)
        Z = dist.sample(200)
        norms = abs(einsum("di,di->i", Z.conj(), Z))
        npt.assert_allclose(norms, ones(200), atol=1e-10)

    def test_sample_concentrated_near_mode(self):
        """With high kappa, |mu^H z|^2 should be close to 1."""
        D = 3
        mu = self._unit_mu(D)
        dist = ComplexWatsonDistribution(mu, 200.0)
        Z = dist.sample(100)
        inner_sq = abs(mu.conj() @ Z) ** 2
        self.assertGreater(inner_sq.mean(), 0.9)

    def test_fit_recovers_parameters(self):
        """fit() on many samples should approximately recover mu and kappa."""
        D = 3
        mu = self._unit_mu(D)
        kappa = 8.0
        dist = ComplexWatsonDistribution(mu, kappa)
        Z = dist.sample(1000)
        dist_hat = ComplexWatsonDistribution.fit(Z)
        # Mode mu is only identified up to a global phase rotation
        ip = abs(dist_hat.mu.conj() @ mu)
        self.assertGreater(ip, 0.99)
        self.assertAlmostEqual(dist_hat.kappa, kappa, delta=2.0)


if __name__ == "__main__":
    unittest.main()
