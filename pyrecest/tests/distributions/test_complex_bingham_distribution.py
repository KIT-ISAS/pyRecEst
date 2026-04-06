import unittest

import numpy as np
import numpy.testing as npt

from pyrecest.distributions import ComplexBinghamDistribution


class TestComplexBinghamDistribution(unittest.TestCase):
    """Tests for ComplexBinghamDistribution."""

    def setUp(self):
        # Simple 2x2 diagonal Hermitian B
        self.B2 = np.diag([-3.0, 0.0]).astype(complex)
        self.cB2 = ComplexBinghamDistribution(self.B2)

        # 3x3 diagonal Hermitian B
        self.B3 = np.diag([-5.0, -2.0, 0.0]).astype(complex)
        self.cB3 = ComplexBinghamDistribution(self.B3)

    def test_constructor_hermitian_check(self):
        """Non-Hermitian matrix should raise AssertionError."""
        with self.assertRaises(AssertionError):
            ComplexBinghamDistribution(np.array([[1.0, 1j], [0.0, 1.0]]))

    def test_log_norm_const_finite(self):
        """log_norm_const must be finite."""
        self.assertTrue(np.isfinite(self.cB2.log_norm_const))
        self.assertTrue(np.isfinite(self.cB3.log_norm_const))

    def test_dim(self):
        self.assertEqual(self.cB2.dim, 2)
        self.assertEqual(self.cB3.dim, 3)

    def test_pdf_normalises_to_one_2d(self):
        """MC check: 2-D pdf integrates to 1 over S^3."""
        rng = np.random.default_rng(12345)
        raw = rng.standard_normal((2, 200_000)) + 1j * rng.standard_normal((2, 200_000))
        Z = raw / np.linalg.norm(raw, axis=0, keepdims=True)
        area = 2.0 * np.pi**2  # surface area of S^3
        mc_integral = np.mean(self.cB2.pdf(Z)) * area
        npt.assert_almost_equal(mc_integral, 1.0, decimal=2)

    def test_pdf_positive(self):
        """pdf must return positive values."""
        z = np.array([1.0, 0.0], dtype=complex)
        self.assertGreater(self.cB2.pdf(z), 0.0)

    def test_pdf_batch_vs_single(self):
        """Vectorised pdf matches point-by-point evaluation."""
        rng = np.random.default_rng(0)
        raw = rng.standard_normal((2, 10)) + 1j * rng.standard_normal((2, 10))
        Z = raw / np.linalg.norm(raw, axis=0, keepdims=True)
        batch = self.cB2.pdf(Z)
        single = np.array([self.cB2.pdf(Z[:, k]) for k in range(10)])
        npt.assert_allclose(batch, single, rtol=1e-10)

    def test_pdf_invariant_to_phase(self):
        """pdf(exp(i*alpha)*z) == pdf(z) for any global phase alpha."""
        z = np.array([0.6 + 0.3j, 0.0], dtype=complex)
        z[1] = np.sqrt(1 - np.abs(z[0]) ** 2)
        p0 = self.cB2.pdf(z)
        for alpha in [0.3, 1.0, np.pi]:
            p1 = self.cB2.pdf(np.exp(1j * alpha) * z)
            npt.assert_almost_equal(p1, p0, decimal=10)

    def test_sample_shape(self):
        """sample returns the right shape."""
        np.random.seed(42)
        S = self.cB2.sample(50)
        self.assertEqual(S.shape, (2, 50))

    def test_sample_unit_norm(self):
        """All samples must lie on the unit sphere."""
        np.random.seed(42)
        S = self.cB2.sample(100)
        norms = np.linalg.norm(S, axis=0)
        npt.assert_allclose(norms, np.ones(100), atol=1e-12)

    def test_sample_3d_unit_norm(self):
        """3-D samples also lie on the unit sphere."""
        np.random.seed(7)
        S = self.cB3.sample(50)
        norms = np.linalg.norm(S, axis=0)
        npt.assert_allclose(norms, np.ones(50), atol=1e-12)

    def test_log_norm_2d_analytic(self):
        a = 3.0
        B = np.diag([-a, 0.0]).astype(complex)
        log_C_expected = np.log(2 * np.pi**2 / a * (1 - np.exp(-a)))
        log_norm_got = ComplexBinghamDistribution.log_norm(B)
        npt.assert_almost_equal(-log_norm_got, log_C_expected, decimal=6)

    def test_log_norm_equal_eigenvalues(self):
        """Equal eigenvalues (uniform) should not raise."""
        B = np.zeros((3, 3), dtype=complex)
        log_norm = ComplexBinghamDistribution.log_norm(B)
        self.assertTrue(np.isfinite(log_norm))

    def test_fit_returns_instance(self):
        """fit() returns a ComplexBinghamDistribution instance."""
        np.random.seed(0)
        Z = self.cB2.sample(200)
        cB_fit = ComplexBinghamDistribution.fit(Z)
        self.assertIsInstance(cB_fit, ComplexBinghamDistribution)

    def test_fit_recovers_eigenvalue_gap(self):
        """Fit recovers the eigenvalue gap (up to additive const)."""
        np.random.seed(0)
        B = np.diag([-10.0, 0.0]).astype(complex)
        cB = ComplexBinghamDistribution(B)
        Z = cB.sample(2000)
        cB_fit = ComplexBinghamDistribution.fit(Z)
        evals_fit = np.sort(np.real(np.linalg.eigvalsh(cB_fit.B)))
        evals_true = np.sort(np.real(np.linalg.eigvalsh(B)))
        gap_true = evals_true[-1] - evals_true[0]
        gap_fit = evals_fit[-1] - evals_fit[0]
        npt.assert_almost_equal(gap_fit, gap_true, decimal=0)

    def test_cauchy_schwarz_zero_for_identical(self):
        """D_CS(p, p) should be 0."""
        d = ComplexBinghamDistribution.cauchy_schwarz_divergence(self.cB2, self.cB2)
        npt.assert_almost_equal(d, 0.0, decimal=6)

    def test_cauchy_schwarz_symmetric(self):
        """D_CS(p, q) == D_CS(q, p)."""
        B3a = np.diag([-5.0, -1.0, 0.0]).astype(complex)
        B3b = np.diag([-3.0, -2.0, 0.0]).astype(complex)
        d_ab = ComplexBinghamDistribution.cauchy_schwarz_divergence(B3a, B3b)
        d_ba = ComplexBinghamDistribution.cauchy_schwarz_divergence(B3b, B3a)
        npt.assert_almost_equal(d_ab, d_ba, decimal=6)

    def test_cauchy_schwarz_nonneg(self):
        """Cauchy-Schwarz divergence must be >= 0."""
        B3a = np.diag([-5.0, -1.0, 0.0]).astype(complex)
        B3b = np.diag([-3.0, -2.0, 0.0]).astype(complex)
        d = ComplexBinghamDistribution.cauchy_schwarz_divergence(B3a, B3b)
        self.assertGreaterEqual(d, -1e-10)


if __name__ == "__main__":
    unittest.main()
