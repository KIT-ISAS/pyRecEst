import unittest

import numpy.testing as npt
from scipy.optimize import brentq
from scipy.special import i0 as besseli0, i1 as besseli1  # pylint: disable=no-name-in-module

import pyrecest.backend
from pyrecest.backend import (
    arctan2,
    arange,
    array,
    array_equal,
    column_stack,
    cos,
    exp,
    linalg,
    meshgrid,
    pi,
    random,
    sin,
    sqrt,
    sum as backend_sum,
    zeros,
)
from pyrecest.distributions.hypertorus.toroidal_von_mises_matrix_distribution import ToroidalVonMisesMatrixDistribution
from pyrecest.distributions.hypertorus.toroidal_fourier_distribution import ToroidalFourierDistribution

class ToroidalVMMatrixDistributionTest(unittest.TestCase):

    def setUp(self):
        self.mu = array([1.7, 2.3])
        self.kappa = array([0.8, 0.3])
        self.A = 0.1 * array([[1, 2], [3, 4]])
        self.tvm = ToroidalVonMisesMatrixDistribution(self.mu, self.kappa, self.A)
        X, Y = meshgrid(arange(1, 7), arange(1, 7))
        self.testpoints = column_stack([X.ravel(), Y.ravel()])
        self.tvm2 = ToroidalVonMisesMatrixDistribution(self.mu + 1, self.kappa + 0.2, linalg.inv(self.A))

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("jax", "pytorch"),  # pylint: disable=no-member
        reason="Only supported for numpy backend",
    )
    def test_sanity_check(self):
        self.assertIsInstance(self.tvm, ToroidalVonMisesMatrixDistribution)
        self.assertTrue(array_equal(self.tvm.mu, self.mu))
        self.assertTrue(array_equal(self.tvm.kappa, self.kappa))
        self.assertTrue(array_equal(self.tvm.A, self.A))

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("jax", "pytorch"),  # pylint: disable=no-member
        reason="Only supported for numpy backend",
    )
    def test_pdf(self):
        def pdf(xs, mu, kappa, A, C):
            if xs.shape[1] > 1:
                results = [pdf(xs[:, i:i + 1], mu, kappa, A, C) for i in range(xs.shape[1])]  # noqa: E203
                return array(results)
            xi = xs[:, 0]
            return C * exp(
                kappa[0] * cos(xi[0] - mu[0]) +
                kappa[1] * cos(xi[1] - mu[1]) +
                array([cos(xi[0] - mu[0]), sin(xi[0] - mu[0])]) @
                A @
                array([cos(xi[1] - mu[1]), sin(xi[1] - mu[1])])
            )

        npt.assert_allclose(self.tvm.pdf(self.testpoints), pdf(self.testpoints.T, self.mu, self.kappa, self.A, self.tvm.C), rtol=1e-10)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("jax", "pytorch"),  # pylint: disable=no-member
        reason="Only supported for numpy backend",
    )
    def test_integral(self):
        self.assertAlmostEqual(self.tvm.integrate(), 1, places=5)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("jax", "pytorch"),  # pylint: disable=no-member
        reason="Only supported for numpy backend",
    )
    def test_trig_moment_numerical(self):
        npt.assert_allclose(self.tvm.trigonometric_moment_numerical(0), array([1, 1]), atol=1e-5)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("jax", "pytorch"),  # pylint: disable=no-member
        reason="Only supported for numpy backend",
    )
    def test_multiply(self):
        tvmMul = self.tvm.multiply(self.tvm2)
        tvmMulSwapped = self.tvm2.multiply(self.tvm)

        C = self.tvm.pdf(array([[0, 0]])) * self.tvm2.pdf(array([[0, 0]])) / tvmMul.pdf(array([[0, 0]]))
        npt.assert_allclose(self.tvm.pdf(self.testpoints) * self.tvm2.pdf(self.testpoints), C * tvmMul.pdf(self.testpoints), rtol=1e-10)
        npt.assert_allclose(self.tvm.pdf(self.testpoints) * self.tvm2.pdf(self.testpoints), C * tvmMulSwapped.pdf(self.testpoints), rtol=1e-10)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("jax", "pytorch"),  # pylint: disable=no-member
        reason="Only supported for numpy backend",
    )
    def test_compare_with_ToroidalFourier_multiplication(self):
        tvmMul = self.tvm.multiply(self.tvm2)
        tvmMulSwapped = self.tvm2.multiply(self.tvm)
        n = (45, 45)
        tf = ToroidalFourierDistribution.from_distribution(self.tvm, n)
        tf2 = ToroidalFourierDistribution.from_distribution(self.tvm2, n)
        tfMul = tf.multiply(tf2)
        tfMulSwapped = tf2.multiply(tf)

        tvmMul.C = 1
        tvmMul.C = 1 / tvmMul.integrate()
        tvmMulSwapped.C = 1
        tvmMulSwapped.C = 1 / tvmMulSwapped.integrate()

        npt.assert_allclose(tfMul.pdf(self.testpoints), tvmMul.pdf(self.testpoints), atol=1e-5)
        npt.assert_allclose(tfMulSwapped.pdf(self.testpoints), tvmMulSwapped.pdf(self.testpoints), atol=1e-5)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("jax", "pytorch"),  # pylint: disable=no-member
        reason="Only supported for numpy backend",
    )
    def test_product_approximation(self):  # pylint: disable=too-many-locals
        mu1 = array([1.7, 0.5])
        kappa1 = array([0.8, 0.3])
        A1 = 0.1 * array([[1, 0.2], [0.2, 1]])
        mu2 = array([1.5, 1])
        kappa2 = array([0.7, 0.2])
        A2 = 0.1 * array([[1, -0.2], [-0.2, 1]])

        tvm1 = ToroidalVonMisesMatrixDistribution(mu1, kappa1, A1)
        tvm2 = ToroidalVonMisesMatrixDistribution(mu2, kappa2, A2)
        tvm3 = tvm1.multiply(tvm2)

        random.seed(1234)
        n = 1e4
        samples = random.uniform(low=0.0, high=2 * pi, size=(2, int(n)))
        weights1 = tvm1.pdf(samples.T)
        weights2 = tvm2.pdf(samples.T)
        weights = weights1 * weights2

        weights = weights / backend_sum(weights)  # normalize; subsequent backend_sum(x * weights) acts as weighted mean

        muApprox = arctan2(
            backend_sum(sin(samples) * weights, axis=1),
            backend_sum(cos(samples) * weights, axis=1),
        )
        rbar = sqrt(
            backend_sum(cos(samples) * weights, axis=1) ** 2 +
            backend_sum(sin(samples) * weights, axis=1) ** 2
        )
        kappaApprox = array([brentq(lambda k, r=r: besseli1(k) / besseli0(k) - r, 1e-9, 50) if r > 1e-9 else 1e-6 for r in rbar])

        Aapprox = zeros((2, 2))
        Aapprox[0, 0] = backend_sum(cos(samples[0, :] - muApprox[0]) * cos(samples[1, :] - muApprox[1]) * weights)
        Aapprox[0, 1] = backend_sum(cos(samples[0, :] - muApprox[0]) * sin(samples[1, :] - muApprox[1]) * weights)
        Aapprox[1, 0] = backend_sum(sin(samples[0, :] - muApprox[0]) * cos(samples[1, :] - muApprox[1]) * weights)
        Aapprox[1, 1] = backend_sum(sin(samples[0, :] - muApprox[0]) * sin(samples[1, :] - muApprox[1]) * weights)

        tvmApprox = ToroidalVonMisesMatrixDistribution(muApprox, kappaApprox, Aapprox)

        npt.assert_allclose(tvmApprox.pdf(self.testpoints), tvm3.pdf(self.testpoints), atol=1e-2, rtol=1e-1)

if __name__ == "__main__":
    unittest.main()
