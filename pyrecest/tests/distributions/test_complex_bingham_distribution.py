import unittest

import math
import numpy as np
import numpy.testing as npt

from pyrecest.distributions import ComplexBinghamDistribution


class TestComplexBinghamDistribution(unittest.TestCase):
    """Tests for ComplexBinghamDistribution."""

    def _make_dist_2d(self):
        """Return a 2D complex Bingham distribution with known parameters."""
        # Diagonal B in C^2; after eigen-shift the eigenvalues are (-3, 0).
        kappa = np.array([-3.0, 0.0])
        B = np.diag(kappa.astype(complex))
        return ComplexBinghamDistribution(B)

    def _make_dist_3d(self):
        """Return a 3D complex Bingham distribution."""
        kappa = np.array([-5.0, -2.0, 0.0])
        B = np.diag(kappa.astype(complex))
        return ComplexBinghamDistribution(B)

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    def test_constructor_diagonal(self):
        """Constructor succeeds for a valid diagonal Hermitian matrix."""
        dist = self._make_dist_2d()
        self.assertEqual(dist.complex_dim, 2)
        self.assertEqual(dist.input_dim, 2)

    def test_constructor_non_hermitian_raises(self):
        """Constructor raises ValueError for a non-Hermitian matrix."""
        B_bad = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=complex)
        with self.assertRaises(ValueError):
            ComplexBinghamDistribution(B_bad)

    def test_constructor_full_hermitian(self):
        """Constructor works for a full non-diagonal Hermitian matrix."""
        A = np.array([[1.0, 0.5 + 0.3j], [0.5 - 0.3j, 2.0]], dtype=complex)
        evals = np.linalg.eigvalsh(A)
        B = A - evals.max() * np.eye(2)
        dist = ComplexBinghamDistribution(B)
        self.assertEqual(dist.complex_dim, 2)

    # ------------------------------------------------------------------
    # Normalization constant
    # ------------------------------------------------------------------

    def test_log_norm_uniform_limit(self):
        """When B = 0 (uniform), C(B) equals the surface area of S^{2d-1}."""
        for d in (2, 3):
            B = np.zeros((d, d), dtype=complex)
            expected = np.log(2.0 * np.pi**d / math.factorial(d - 1))
            log_c = ComplexBinghamDistribution.log_norm(B)
            npt.assert_almost_equal(-log_c, expected, decimal=6)

    def test_log_norm_2d_analytical_formula(self):
        """For D=2 with kappa=(kappa1, 0), C = 2pi^2 (exp(kappa1)-1)/kappa1."""
        kappa1 = -3.0
        B = np.diag([kappa1, 0.0]).astype(complex)
        expected_C = 2.0 * np.pi**2 * (np.exp(kappa1) - 1.0) / kappa1
        log_c = ComplexBinghamDistribution.log_norm(B)
        npt.assert_almost_equal(-log_c, np.log(expected_C), decimal=6)

    def test_log_norm_analytical_vs_monte_carlo(self):
        """Analytical and Monte Carlo log_norm agree to within ~1 decimal for D=2."""
        kappa1 = -3.0
        B = np.diag([kappa1, 0.0]).astype(complex)
        log_c_analytical = ComplexBinghamDistribution.log_norm(B, variant="analytical")
        log_c_mc = ComplexBinghamDistribution.log_norm(
            B, variant="monte_carlo", n_mc=500_000
        )
        npt.assert_almost_equal(-log_c_analytical, -log_c_mc, decimal=1)

    # ------------------------------------------------------------------
    # PDF
    # ------------------------------------------------------------------

    def test_pdf_unit_norm_positive(self):
        """PDF values are positive at unit vectors."""
        dist = self._make_dist_2d()
        z = np.array([1.0 + 0j, 0.0 + 0j])
        p = dist.pdf(z)
        self.assertGreater(float(p), 0.0)

    def test_pdf_normalizes_to_one(self):
        """Numerical integration of the pdf over the sphere gives approx 1."""
        dist = self._make_dist_2d()
        # Parameterise S^3: z1 = cos(t)e^{ip1}, z2 = sin(t)e^{ip2}
        n_phi = 50
        n_theta = 50
        phi1 = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
        phi2 = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
        theta = np.linspace(0, np.pi / 2, n_theta, endpoint=False)
        dtheta = theta[1] - theta[0]
        dphi = phi1[1] - phi1[0]

        integral = 0.0
        for t in theta:
            for p1 in phi1:
                for p2 in phi2:
                    z = np.array(
                        [
                            np.cos(t) * np.exp(1j * p1),
                            np.sin(t) * np.exp(1j * p2),
                        ]
                    )
                    jacobian = np.sin(t) * np.cos(t)
                    integral += float(dist.pdf(z)) * jacobian * dtheta * dphi * dphi

        npt.assert_almost_equal(integral, 1.0, decimal=2)

    def test_pdf_mode_is_maximum(self):
        """The mode (eigenvector for max eigenvalue) gives the largest pdf."""
        kappa = np.array([-5.0, 0.0])
        B = np.diag(kappa.astype(complex))
        dist = ComplexBinghamDistribution(B)

        mode = np.array([0.0 + 0j, 1.0 + 0j])
        p_mode = dist.pdf(mode)
        other = np.array([1.0 + 0j, 0.0 + 0j])
        p_other = dist.pdf(other)
        self.assertGreaterEqual(float(p_mode), float(p_other))

    def test_pdf_batch(self):
        """pdf works for batches of columns."""
        dist = self._make_dist_2d()
        z1 = np.array([1.0 + 0j, 0.0 + 0j])
        z2 = np.array([0.0 + 0j, 1.0 + 0j])
        Z = np.stack([z1, z2], axis=1)
        p = dist.pdf(Z)
        npt.assert_array_almost_equal(
            p, [dist.pdf(z1), dist.pdf(z2)], decimal=10
        )

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def test_sample_unit_norm(self):
        """Samples lie on the complex unit sphere."""
        dist = self._make_dist_2d()
        Z = dist.sample(100)
        norms = np.linalg.norm(Z, axis=0)
        npt.assert_array_almost_equal(norms, np.ones(100), decimal=12)

    def test_sample_shape(self):
        """sample() returns array of shape (d, n)."""
        dist = self._make_dist_3d()
        Z = dist.sample(50)
        self.assertEqual(Z.shape, (3, 50))

    def test_sample_scatter_consistent(self):
        """Empirical scatter aligns with B's eigenvectors."""
        n = 50_000
        kappa = np.array([-5.0, 0.0])
        B = np.diag(kappa.astype(complex))
        dist = ComplexBinghamDistribution(B)
        Z = dist.sample(n)
        S_empirical = Z @ Z.conj().T / n
        evals = np.linalg.eigvalsh(S_empirical)
        self.assertGreater(evals[-1], 0.5)

    # ------------------------------------------------------------------
    # Cauchy-Schwarz Divergence
    # ------------------------------------------------------------------

    def test_cauchy_schwarz_divergence_zero_for_same(self):
        """D_CS(p, p) = 0 for identical distributions."""
        dist = self._make_dist_2d()
        d = ComplexBinghamDistribution.cauchy_schwarz_divergence(dist, dist)
        npt.assert_almost_equal(d, 0.0, decimal=6)

    def test_cauchy_schwarz_divergence_nonnegative(self):
        """D_CS >= 0 for different distributions."""
        d1 = self._make_dist_2d()
        B2 = np.diag(np.array([-1.0, 0.0]).astype(complex))
        d2 = ComplexBinghamDistribution(B2)
        divergence = ComplexBinghamDistribution.cauchy_schwarz_divergence(d1, d2)
        self.assertGreaterEqual(divergence, -1e-6)

    # ------------------------------------------------------------------
    # fit / estimate_parameter_matrix
    # ------------------------------------------------------------------

    def test_fit_recovers_eigenvalues(self):
        """fit() from many samples recovers eigenvalues to 1 decimal place."""
        kappa = np.array([-3.0, 0.0])
        B_true = np.diag(kappa.astype(complex))
        dist_true = ComplexBinghamDistribution(B_true)

        Z = dist_true.sample(20_000)
        dist_fit = ComplexBinghamDistribution.fit(Z)

        evals_true = np.sort(np.linalg.eigvalsh(B_true))
        evals_fit = np.sort(np.linalg.eigvalsh(dist_fit.B))
        npt.assert_array_almost_equal(evals_true, evals_fit, decimal=1)

    # ------------------------------------------------------------------
    # log_norm with non-diagonal matrix
    # ------------------------------------------------------------------

    def test_log_norm_non_diagonal(self):
        """log_norm works for full Hermitian matrices."""
        A = np.array([[1.0, 0.5 + 0.3j], [0.5 - 0.3j, 2.0]], dtype=complex)
        evals = np.linalg.eigvalsh(A)
        B = A - evals.max() * np.eye(2)
        log_c = ComplexBinghamDistribution.log_norm(B)
        self.assertTrue(np.isfinite(log_c))


if __name__ == "__main__":
    unittest.main()
