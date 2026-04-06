import unittest

import numpy as np
import numpy.testing as npt
import pyrecest.backend

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array
from pyrecest.distributions import ComplexAngularCentralGaussianDistribution


class TestComplexAngularCentralGaussianDistribution(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        # Identity matrix case (uniform distribution on complex unit sphere)
        self.C_identity_2d = array(np.eye(2, dtype=complex))
        self.dist_identity_2d = ComplexAngularCentralGaussianDistribution(
            self.C_identity_2d
        )

        # Non-trivial Hermitian positive definite matrix for 2D case
        # C = [[2, 1+1j], [1-1j, 3]]
        C_vals = np.array([[2.0, 1.0 + 1.0j], [1.0 - 1.0j, 3.0]])
        self.C_nontrivial_2d = array(C_vals)
        self.dist_nontrivial_2d = ComplexAngularCentralGaussianDistribution(
            self.C_nontrivial_2d
        )

    def test_constructor_valid(self):
        """Test that constructor accepts a Hermitian matrix."""
        self.assertEqual(self.dist_identity_2d.dim, 2)
        self.assertEqual(self.dist_nontrivial_2d.dim, 2)

    def test_constructor_non_hermitian_raises(self):
        """Test that constructor rejects a non-Hermitian matrix."""
        C_bad = array(np.array([[1.0, 2.0 + 1.0j], [0.0, 1.0]]))
        with self.assertRaises(AssertionError):
            ComplexAngularCentralGaussianDistribution(C_bad)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on JAX backend",
    )  # pylint: disable=no-member
    def test_pdf_identity_uniform(self):
        """For C=I, the pdf should be constant gamma(d)/(2*pi^d) on the unit sphere."""
        d = 2
        # Expected: gamma(2) / (2*pi^2) = 1 / (2*pi^2)
        expected = 1.0 / (2.0 * np.pi**d)  # gamma(2)=1

        # Test on several unit vectors
        z1 = array(np.array([[1.0, 0.0]], dtype=complex))
        z2 = array(np.array([[0.0, 1.0]], dtype=complex))
        z3 = array((np.array([1.0, 1.0j]) / np.sqrt(2.0)).reshape(1, -1))
        z4 = array((np.array([1.0 + 1.0j, 1.0 - 1.0j]) / 2.0).reshape(1, -1))

        for z in [z1, z2, z3, z4]:
            p = self.dist_identity_2d.pdf(z)
            npt.assert_allclose(
                float(np.real(np.array(p[0]))),
                expected,
                rtol=1e-6,
                err_msg=f"PDF for identity C is not constant at {z}",
            )

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on JAX backend",
    )  # pylint: disable=no-member
    def test_pdf_positive(self):
        """PDF values should be positive for any unit vector."""
        z = array(np.array([[1.0 / np.sqrt(2.0), 1.0j / np.sqrt(2.0)]]))
        p = self.dist_nontrivial_2d.pdf(z)
        self.assertGreater(float(np.real(np.array(p[0]))), 0.0)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on JAX backend",
    )  # pylint: disable=no-member
    def test_pdf_batch_vs_single(self):
        """Batch PDF evaluation should match individual evaluations."""
        zs = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0 / np.sqrt(2.0), 1.0j / np.sqrt(2.0)],
            ],
            dtype=complex,
        )
        za = array(zs)

        p_batch = self.dist_nontrivial_2d.pdf(za)
        for i, z in enumerate(zs):
            p_single = self.dist_nontrivial_2d.pdf(array(z.reshape(1, -1)))
            npt.assert_allclose(
                float(np.real(np.array(p_batch[i]))),
                float(np.real(np.array(p_single[0]))),
                rtol=1e-10,
            )

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on JAX backend",
    )  # pylint: disable=no-member
    def test_sample_unit_norm(self):
        """Sampled vectors should lie on the complex unit sphere."""
        n = 100
        Z = self.dist_nontrivial_2d.sample(n)
        Z_np = np.array(Z)
        norms_sq = np.array(
            [np.real(np.sum(Z_np[k] * np.conj(Z_np[k]))) for k in range(n)]
        )
        npt.assert_allclose(norms_sq, np.ones(n), atol=1e-10)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on JAX backend",
    )  # pylint: disable=no-member
    def test_sample_correct_dim(self):
        """Sampled vectors should have the correct shape."""
        n = 50
        Z = self.dist_identity_2d.sample(n)
        self.assertEqual(Z.shape[0], n)
        self.assertEqual(Z.shape[1], 2)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on JAX backend",
    )  # pylint: disable=no-member
    def test_estimate_parameter_matrix_identity(self):
        """Fitting samples from identity-C distribution should recover approx identity."""
        pyrecest.backend.random.seed(42)  # pylint: disable=no-member
        n = 2000
        Z = self.dist_identity_2d.sample(n)
        C_est = ComplexAngularCentralGaussianDistribution.estimate_parameter_matrix(
            Z, n_iterations=100
        )
        # Normalize C_est to have trace equal to 2 (matching identity)
        C_est_np = np.array(C_est)
        C_est_normalized = C_est_np / np.trace(C_est_np).real * 2.0
        npt.assert_allclose(
            np.real(C_est_normalized),
            np.eye(2),
            atol=0.15,
            err_msg="Estimated C does not approximately match identity",
        )

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on JAX backend",
    )  # pylint: disable=no-member
    def test_fit_returns_distribution(self):
        """fit() should return a ComplexAngularCentralGaussianDistribution."""
        pyrecest.backend.random.seed(0)  # pylint: disable=no-member
        Z = self.dist_identity_2d.sample(50)
        dist = ComplexAngularCentralGaussianDistribution.fit(Z, n_iterations=10)
        self.assertIsInstance(dist, ComplexAngularCentralGaussianDistribution)
        self.assertEqual(dist.dim, 2)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on JAX backend",
    )  # pylint: disable=no-member
    def test_3d_case(self):
        """Test basic functionality for d=3."""
        C_3d = array(np.eye(3, dtype=complex))
        dist = ComplexAngularCentralGaussianDistribution(C_3d)
        self.assertEqual(dist.dim, 3)

        Z = dist.sample(20)
        self.assertEqual(Z.shape, (20, 3))

        # Check unit norms
        Z_np = np.array(Z)
        norms_sq = np.array(
            [np.real(np.sum(Z_np[k] * np.conj(Z_np[k]))) for k in range(20)]
        )
        npt.assert_allclose(norms_sq, np.ones(20), atol=1e-10)

        # For d=3, C=I: pdf should be gamma(3)/(2*pi^3) = 2/(2*pi^3) = 1/pi^3
        z_test = array(np.array([[1.0, 0.0, 0.0]], dtype=complex))
        p = dist.pdf(z_test)
        expected = 1.0 / np.pi**3  # gamma(3)=2!, 2/(2*pi^3)=1/pi^3
        npt.assert_allclose(float(np.real(np.array(p[0]))), expected, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
