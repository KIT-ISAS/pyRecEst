import unittest

import numpy as np
import numpy.testing as npt

from pyrecest.distributions.hypersphere_subset.complex_watson_distribution import (
    ComplexWatsonDistribution,
)
from pyrecest.distributions.hypersphere_subset.bayesian_complex_watson_mixture_model import (
    _simplex_integral,
)


def _random_unit_vector(D, rng=None):
    """Return a random unit vector in C^D."""
    rng = rng or np.random.default_rng(42)
    z = rng.standard_normal(D) + 1j * rng.standard_normal(D)
    return z / np.linalg.norm(z)


class TestComplexWatsonLogNorm(unittest.TestCase):
    """Tests for the log normalisation constant."""

    def test_scalar_input(self):
        val = ComplexWatsonDistribution.log_norm(3, 5.0)
        self.assertIsInstance(val, float)

    def test_array_input(self):
        kappas = np.array([0.1, 1.0, 10.0, 200.0])
        vals = ComplexWatsonDistribution.log_norm(3, kappas)
        self.assertEqual(vals.shape, (4,))

    def test_D2_low_kappa(self):
        # For low kappa, log C ~ log(2) + 2*log(pi) - log(Gamma(2)) + log(1) = log(2 pi^2)
        # so log_c = -log(2*pi^2)
        log_c = ComplexWatsonDistribution.log_norm(2, 1e-10)
        expected = -np.log(2 * np.pi**2)
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
        self.rng = np.random.default_rng(0)

    def _unit_mu(self, D):
        z = self.rng.standard_normal(D) + 1j * self.rng.standard_normal(D)
        return z / np.linalg.norm(z)

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
        Z = np.stack([_random_unit_vector(D, self.rng) for _ in range(10)], axis=1)
        p = dist.pdf(Z)
        self.assertTrue(np.all(p > 0))

    def test_pdf_mode_is_maximum(self):
        """The PDF at the mode (mu) should be >= the PDF at any other point."""
        D = 3
        mu = self._unit_mu(D)
        dist = ComplexWatsonDistribution(mu, 10.0)
        Z = np.stack([_random_unit_vector(D, self.rng) for _ in range(50)], axis=1)
        p_mode = dist.pdf(mu.reshape(-1, 1))[0]
        self.assertTrue(np.all(p_mode >= dist.pdf(Z) - 1e-10))

    def test_pdf_antipodal_symmetry(self):
        """The PDF should be the same at z and -z (|mu^H z|^2 = |mu^H (-z)|^2)."""
        D = 3
        mu = self._unit_mu(D)
        dist = ComplexWatsonDistribution(mu, 5.0)
        z = _random_unit_vector(D, self.rng).reshape(-1, 1)
        npt.assert_allclose(dist.pdf(z), dist.pdf(-z), rtol=1e-10)

    def test_pdf_phase_invariance(self):
        """Multiplying z by a complex phase should not change the PDF."""
        D = 3
        mu = self._unit_mu(D)
        dist = ComplexWatsonDistribution(mu, 5.0)
        z = _random_unit_vector(D, self.rng).reshape(-1, 1)
        phase = np.exp(1j * 1.23)
        npt.assert_allclose(dist.pdf(z), dist.pdf(phase * z), rtol=1e-10)

    def test_hypergeometric_ratio_bounds(self):
        D = 4
        r0 = ComplexWatsonDistribution._hypergeometric_ratio(0.0, D)
        r_large = ComplexWatsonDistribution._hypergeometric_ratio(1000.0, D)
        self.assertAlmostEqual(r0, 1.0 / D, places=5)
        self.assertGreater(r_large, 0.99)
        self.assertLessEqual(r_large, 1.0)

    def test_hypergeometric_ratio_inverse(self):
        D = 3
        for kappa in [0.5, 5.0, 50.0]:
            r = ComplexWatsonDistribution._hypergeometric_ratio(kappa, D)
            kappa_hat = ComplexWatsonDistribution._hypergeometric_ratio_inverse(r, D)
            self.assertAlmostEqual(kappa_hat, kappa, places=3)

    def test_sample_on_unit_sphere(self):
        D = 3
        mu = self._unit_mu(D)
        dist = ComplexWatsonDistribution(mu, 5.0)
        Z = dist.sample(200)
        norms = np.abs(np.einsum("di,di->i", Z.conj(), Z))
        npt.assert_allclose(norms, np.ones(200), atol=1e-10)

    def test_sample_concentrated_near_mode(self):
        """With high kappa, |mu^H z|^2 should be close to 1."""
        D = 3
        mu = self._unit_mu(D)
        dist = ComplexWatsonDistribution(mu, 200.0)
        Z = dist.sample(100)
        inner_sq = np.abs(mu.conj() @ Z) ** 2
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


class TestSimplexIntegral(unittest.TestCase):
    def test_D1(self):
        self.assertAlmostEqual(_simplex_integral(np.array([3.0])), np.exp(3.0))

    def test_D2_known(self):
        # int_0^1 exp(a*t + b*(1-t)) dt = (exp(a) - exp(b)) / (a - b)
        a, b = 2.0, 1.0
        expected = (np.exp(a) - np.exp(b)) / (a - b)
        result = _simplex_integral(np.array([a, b]))
        self.assertAlmostEqual(result, expected, places=8)

    def test_D3_nonnegative(self):
        result = _simplex_integral(np.array([2.0, 1.0, 0.0]))
        self.assertGreater(result, 0.0)
        # Known value from direct integration: exp(2)/2 - exp(1) + 1/2
        expected = np.exp(2) / 2 - np.exp(1) + 0.5
        self.assertAlmostEqual(result, expected, places=5)


if __name__ == "__main__":
    unittest.main()
