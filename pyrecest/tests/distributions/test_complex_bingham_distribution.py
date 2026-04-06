import unittest

import numpy as np
import numpy.testing as npt
import pyrecest.backend

from pyrecest.distributions import ComplexBinghamDistribution


class TestComplexBinghamDistribution(unittest.TestCase):
    """Tests for ComplexBinghamDistribution."""

    def _make_diagonal_dist(self):
        """2-D diagonal complex Bingham distribution for simple tests."""
        B = np.diag([-3.0, 0.0]).astype(complex)
        return ComplexBinghamDistribution(B)

    def _make_full_dist(self):
        """3-D complex Bingham distribution with a full Hermitian B."""
        rng = np.random.default_rng(42)
        A = rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))
        B = -(A @ A.conj().T)  # negative definite -> mode is well-defined
        return ComplexBinghamDistribution(B)

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    def test_constructor_raises_on_non_hermitian(self):
        B = np.array([[1.0, 1j], [0.0, -1.0]])  # not Hermitian
        with self.assertRaises(ValueError):
            ComplexBinghamDistribution(B)

    def test_constructor_diagonal(self):
        cB = self._make_diagonal_dist()
        self.assertEqual(cB.d, 2)
        # underlying real sphere is S^{2*2-1} = S^3 => sphere dim = 2
        self.assertEqual(cB.dim, 2)

    # ------------------------------------------------------------------
    # Normalisation constant
    # ------------------------------------------------------------------

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on JAX backend",
    )  # pylint: disable=no-member
    def test_log_norm_diagonal_d2(self):
        """log_norm returns -log(integral), should be finite."""
        B = np.diag([-5.0, 0.0]).astype(complex)
        log_c = ComplexBinghamDistribution.log_norm(B)
        self.assertTrue(np.isfinite(log_c))
        # The integral over S^3 is > surface_area implies log_c is finite
        self.assertTrue(np.isfinite(log_c))

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on JAX backend",
    )  # pylint: disable=no-member
    def test_log_norm_uniform_case(self):
        """Near-zero B => log_norm ≈ -log(surface area of S^{2d-1})."""
        B = np.zeros((2, 2), dtype=complex)
        log_c = ComplexBinghamDistribution.log_norm(B)
        expected = -np.log(ComplexBinghamDistribution.unit_sphere_surface(2))
        npt.assert_almost_equal(log_c, expected, decimal=6)

    # ------------------------------------------------------------------
    # PDF
    # ------------------------------------------------------------------

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on JAX backend",
    )  # pylint: disable=no-member
    def test_pdf_normalises_to_one(self):
        """Monte Carlo integral of the pdf over the complex unit sphere ≈ 1."""
        cB = self._make_diagonal_dist()
        integral = cB.integral(n_samples=500_000)
        npt.assert_almost_equal(integral, 1.0, decimal=2)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on JAX backend",
    )  # pylint: disable=no-member
    def test_pdf_mode_is_maximum(self):
        """The pdf at the mode should be >= pdf at random points."""
        B = np.diag([-10.0, 0.0]).astype(complex)
        cB = ComplexBinghamDistribution(B)
        # Mode is the eigenvector for eigenvalue 0 = [0, 1]
        mode = np.array([0.0, 1.0], dtype=complex)
        p_mode = cB.pdf(mode)
        rng = np.random.default_rng(0)
        z = rng.standard_normal((2, 100)) + 1j * rng.standard_normal((2, 100))
        z /= np.sqrt(np.sum(np.abs(z) ** 2, axis=0, keepdims=True))
        p_random = cB.pdf(z)
        self.assertTrue(np.all(p_mode >= p_random))

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on JAX backend",
    )  # pylint: disable=no-member
    def test_pdf_symmetry(self):
        """pdf(z) == pdf(-z) (antipodal symmetry)."""
        cB = self._make_diagonal_dist()
        rng = np.random.default_rng(7)
        z = rng.standard_normal(2) + 1j * rng.standard_normal(2)
        z /= np.linalg.norm(z)
        npt.assert_almost_equal(cB.pdf(z), cB.pdf(-z), decimal=10)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on JAX backend",
    )  # pylint: disable=no-member
    def test_sample_unit_norm(self):
        """Samples should lie on the complex unit sphere."""
        cB = self._make_diagonal_dist()
        samples = cB.sample(50)
        norms = np.sqrt(np.sum(np.abs(samples) ** 2, axis=0))
        npt.assert_array_almost_equal(norms, np.ones(50), decimal=10)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on JAX backend",
    )  # pylint: disable=no-member
    def test_sample_shape(self):
        cB = self._make_diagonal_dist()
        samples = cB.sample(30)
        self.assertEqual(samples.shape, (2, 30))

    # ------------------------------------------------------------------
    # Cauchy-Schwarz divergence
    # ------------------------------------------------------------------

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on JAX backend",
    )  # pylint: disable=no-member
    def test_cauchy_schwarz_divergence_self_is_zero(self):
        """CSD of a distribution with itself should be 0."""
        cB = self._make_diagonal_dist()
        div = ComplexBinghamDistribution.cauchy_schwarz_divergence(cB, cB)
        npt.assert_almost_equal(div, 0.0, decimal=6)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on JAX backend",
    )  # pylint: disable=no-member
    def test_cauchy_schwarz_divergence_non_negative(self):
        B1 = np.diag([-3.0, 0.0]).astype(complex)
        B2 = np.diag([-5.0, 0.0]).astype(complex)
        div = ComplexBinghamDistribution.cauchy_schwarz_divergence(
            ComplexBinghamDistribution(B1), ComplexBinghamDistribution(B2)
        )
        self.assertGreaterEqual(div, -1e-8)

    # ------------------------------------------------------------------
    # unit_sphere_surface
    # ------------------------------------------------------------------

    def test_unit_sphere_surface_d1(self):
        # C^1 sphere = S^1 = circle, area = 2*pi
        npt.assert_almost_equal(ComplexBinghamDistribution.unit_sphere_surface(1), 2.0 * np.pi)

    def test_unit_sphere_surface_d2(self):
        # C^2 sphere = S^3, area = 2*pi^2
        npt.assert_almost_equal(
            ComplexBinghamDistribution.unit_sphere_surface(2), 2.0 * np.pi**2
        )


if __name__ == "__main__":
    unittest.main()
