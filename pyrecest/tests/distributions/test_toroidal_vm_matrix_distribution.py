import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
import pyrecets.backend
from pyrecets.backend import array, mod, pi
from pyrecets.distributions.hypertorus.toroidal_vm_matrix_distribution import (
    ToroidalVMMatrixDistribution,
)


class TestToroidalVMMatrixDistribution(unittest.TestCase):
    def setUp(self):
        self.mu = array([1.0, 2.0])
        self.kappa = array([0.5, 0.7])
        self.A = array([[0.3, 0.1], [-0.2, 0.4]])
        self.tvm = ToroidalVMMatrixDistribution(self.mu, self.kappa, self.A)

    def test_instance(self):
        self.assertIsInstance(self.tvm, ToroidalVMMatrixDistribution)

    def test_properties(self):
        npt.assert_allclose(self.tvm.mu, self.mu, atol=1e-10)
        npt.assert_allclose(self.tvm.kappa, self.kappa, atol=1e-10)
        npt.assert_allclose(self.tvm.A, self.A, atol=1e-10)

    def test_pdf_positive(self):
        xs = array([[0.5, 1.0], [1.0, 2.0], [3.0, 4.0]])
        vals = self.tvm.pdf(xs)
        for v in vals.ravel():
            self.assertGreater(float(v), 0.0)

    def test_mu_wrapped(self):
        mu_unwrapped = array([1.0 + 2 * float(pi), 2.0])
        tvm2 = ToroidalVMMatrixDistribution(mu_unwrapped, self.kappa, self.A)
        npt.assert_allclose(tvm2.mu, self.tvm.mu, atol=1e-10)

    @unittest.skipIf(
        pyrecets.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_integral(self):
        self.assertAlmostEqual(self.tvm.integrate(), 1.0, delta=1e-4)

    @unittest.skipIf(
        pyrecets.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_integral_numerical_normalization(self):
        # High concentration forces numerical normalization
        tvm_high = ToroidalVMMatrixDistribution(
            array([0.5, 1.0]), array([2.0, 2.0]), array([[0.1, 0.0], [0.0, 0.1]])
        )
        self.assertAlmostEqual(tvm_high.integrate(), 1.0, delta=1e-4)

    @unittest.skipIf(
        pyrecets.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_multiply_integrates_to_1(self):
        tvm2 = ToroidalVMMatrixDistribution(
            array([0.3, 1.5]), array([0.4, 0.6]), array([[0.1, 0.0], [0.0, 0.2]])
        )
        product = self.tvm.multiply(tvm2)
        self.assertIsInstance(product, ToroidalVMMatrixDistribution)
        self.assertAlmostEqual(product.integrate(), 1.0, delta=2e-3)

    @unittest.skipIf(
        pyrecets.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_marginalize_to_1d_dim0(self):
        marginal = self.tvm.marginalize_to_1d(0)
        self.assertAlmostEqual(marginal.integrate(), 1.0, delta=1e-4)

    @unittest.skipIf(
        pyrecets.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_marginalize_to_1d_dim1(self):
        marginal = self.tvm.marginalize_to_1d(1)
        self.assertAlmostEqual(marginal.integrate(), 1.0, delta=1e-4)

    def test_shift(self):
        shift = array([0.5, -0.3])
        shifted = self.tvm.shift(shift)
        expected_mu = mod(self.mu + shift, 2.0 * float(pi))
        npt.assert_allclose(shifted.mu, expected_mu, atol=1e-10)
        # A and kappa unchanged
        npt.assert_allclose(shifted.kappa, self.tvm.kappa, atol=1e-10)
        npt.assert_allclose(shifted.A, self.tvm.A, atol=1e-10)

    def test_shift_does_not_modify_original(self):
        original_mu = array(self.tvm.mu)
        _ = self.tvm.shift(array([1.0, 1.0]))
        npt.assert_allclose(self.tvm.mu, original_mu, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
