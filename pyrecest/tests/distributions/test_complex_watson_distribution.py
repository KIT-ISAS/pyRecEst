import unittest

import numpy as np
import numpy.testing as npt
import pyrecest.backend

from pyrecest.distributions import ComplexWatsonDistribution


class TestComplexWatsonDistribution(unittest.TestCase):
    def setUp(self):
        # 2-D complex unit vector
        self.mu2 = np.array([1.0, 0.0], dtype=complex)
        self.kappa2 = 2.0

        # 3-D complex unit vector
        raw = np.array([1.0, 1j, 1.0 + 1j], dtype=complex)
        self.mu3 = raw / np.linalg.norm(raw)
        self.kappa3 = 5.0

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------
    def test_constructor_stores_attributes(self):
        cw = ComplexWatsonDistribution(self.mu2, self.kappa2)
        npt.assert_array_equal(cw.mu, self.mu2)
        self.assertEqual(cw.kappa, self.kappa2)
        self.assertEqual(cw.dim, 2)

    def test_constructor_rejects_unnormalized_mu(self):
        with self.assertRaises(AssertionError):
            ComplexWatsonDistribution(np.array([1.0, 1.0], dtype=complex), 1.0)

    def test_constructor_rejects_2d_mu(self):
        with self.assertRaises(AssertionError):
            ComplexWatsonDistribution(np.array([[1.0, 0.0]], dtype=complex), 1.0)

    # ------------------------------------------------------------------
    # mean
    # ------------------------------------------------------------------
    def test_mean_returns_mu(self):
        cw = ComplexWatsonDistribution(self.mu2, self.kappa2)
        npt.assert_array_equal(cw.mean(), self.mu2)

    # ------------------------------------------------------------------
    # log_norm
    # ------------------------------------------------------------------
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_log_norm_scalar(self):
        lc = ComplexWatsonDistribution.log_norm(2, 2.0)
        self.assertTrue(np.isfinite(lc))
        self.assertIsInstance(lc, float)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_log_norm_array(self):
        kappas = np.array([0.0, 0.1, 1.0, 10.0, 200.0])
        lcs = ComplexWatsonDistribution.log_norm(3, kappas)
        self.assertEqual(lcs.shape, kappas.shape)
        self.assertTrue(np.all(np.isfinite(np.asarray(lcs, dtype=float))))

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_log_norm_consistency_across_regimes(self):
        D = 3
        kappa_border = 1.0 / D
        delta = 1e-4
        lc_lo = ComplexWatsonDistribution.log_norm(D, kappa_border - delta)
        lc_me = ComplexWatsonDistribution.log_norm(D, kappa_border + delta)
        self.assertAlmostEqual(lc_lo, lc_me, places=2)

        lc_me2 = ComplexWatsonDistribution.log_norm(D, 100.0 - 1e-6)
        lc_hi = ComplexWatsonDistribution.log_norm(D, 100.0 + 1e-6)
        self.assertAlmostEqual(lc_me2, lc_hi, places=2)

    # ------------------------------------------------------------------
    # pdf
    # ------------------------------------------------------------------
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_pdf_nonnegative(self):
        cw = ComplexWatsonDistribution(self.mu2, self.kappa2)
        pts = np.array([[1.0, 0.0], [0.0, 1.0], [1j, 0.0]], dtype=complex)
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        p = cw.pdf(pts)
        self.assertTrue(np.all(np.asarray(p, dtype=float) >= 0.0))

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_pdf_mode_is_maximum(self):
        cw = ComplexWatsonDistribution(self.mu2, self.kappa2)
        p_mode = cw.pdf(self.mu2)
        other = np.array([0.0, 1.0], dtype=complex)
        p_other = cw.pdf(other)
        self.assertGreaterEqual(p_mode, p_other)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_pdf_antipodal_symmetry(self):
        cw = ComplexWatsonDistribution(self.mu3, self.kappa3)
        raw = np.array([1.0, -1j, 0.5], dtype=complex)
        z = raw / np.linalg.norm(raw)
        npt.assert_allclose(cw.pdf(z), cw.pdf(-z), rtol=1e-12)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_pdf_phase_invariance(self):
        cw = ComplexWatsonDistribution(self.mu3, self.kappa3)
        raw = np.array([1.0, 0.5, -0.5j], dtype=complex)
        z = raw / np.linalg.norm(raw)
        phase = np.exp(1j * 0.7)
        npt.assert_allclose(cw.pdf(z), cw.pdf(phase * z), rtol=1e-12)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_pdf_single_point(self):
        cw = ComplexWatsonDistribution(self.mu2, self.kappa2)
        p = cw.pdf(self.mu2)
        self.assertIsInstance(float(p), float)

    # ------------------------------------------------------------------
    # sample
    # ------------------------------------------------------------------
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_sample_shape(self):
        cw = ComplexWatsonDistribution(self.mu2, self.kappa2)
        np.random.seed(0)
        samples = cw.sample(50)
        self.assertEqual(samples.shape, (50, 2))

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_sample_unit_norm(self):
        cw = ComplexWatsonDistribution(self.mu2, self.kappa2)
        np.random.seed(42)
        samples = cw.sample(200)
        norms = np.linalg.norm(samples, axis=1)
        npt.assert_allclose(norms, np.ones(200), atol=1e-10)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_sample_3d_unit_norm(self):
        cw = ComplexWatsonDistribution(self.mu3, self.kappa3)
        np.random.seed(7)
        samples = cw.sample(100)
        norms = np.linalg.norm(samples, axis=1)
        npt.assert_allclose(norms, np.ones(100), atol=1e-10)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_sample_raises_for_dim1(self):
        mu1 = np.array([1.0 + 0j])
        with self.assertRaises(ValueError):
            ComplexWatsonDistribution(mu1, 2.0).sample(10)

    # ------------------------------------------------------------------
    # estimate_parameters / fit
    # ------------------------------------------------------------------
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_estimate_parameters_recovers_mu(self):
        cw_true = ComplexWatsonDistribution(self.mu2, self.kappa2)
        np.random.seed(0)
        Z = cw_true.sample(500)
        mu_hat, _ = ComplexWatsonDistribution.estimate_parameters(Z)
        overlap = abs(np.vdot(mu_hat, self.mu2))
        self.assertGreater(overlap, 0.9)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_estimate_parameters_recovers_kappa(self):
        cw_true = ComplexWatsonDistribution(self.mu2, self.kappa2)
        np.random.seed(1)
        Z = cw_true.sample(1000)
        _, kappa_hat = ComplexWatsonDistribution.estimate_parameters(Z)
        self.assertAlmostEqual(kappa_hat, self.kappa2, delta=1.0)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_fit_returns_distribution(self):
        cw_true = ComplexWatsonDistribution(self.mu2, self.kappa2)
        np.random.seed(2)
        Z = cw_true.sample(200)
        cw_fit = ComplexWatsonDistribution.fit(Z)
        self.assertIsInstance(cw_fit, ComplexWatsonDistribution)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_fit_with_weights(self):
        cw_true = ComplexWatsonDistribution(self.mu2, self.kappa2)
        np.random.seed(3)
        Z = cw_true.sample(200)
        w = np.ones(200)
        cw_fit = ComplexWatsonDistribution.fit(Z, weights=w)
        self.assertIsInstance(cw_fit, ComplexWatsonDistribution)


if __name__ == "__main__":
    unittest.main()
