import unittest

import numpy as np
from parameterized import parameterized
from pyrecest.distributions.hypertorus.toroidal_von_mises_sine_distribution import (
    ToroidalVonMisesSineDistribution,
)


class ToroidalVMSineDistributionTest(unittest.TestCase):
    def setUp(self):
        self.mu = np.array([1, 2])
        self.kappa = np.array([0.7, 1.4])
        self.lambda_ = 0.5
        self.tvm = ToroidalVonMisesSineDistribution(self.mu, self.kappa, self.lambda_)

    def test_instance(self):
        # sanity check
        self.assertIsInstance(self.tvm, ToroidalVonMisesSineDistribution)

    def test_mu_kappa_lambda(self):
        np.testing.assert_almost_equal(self.tvm.mu, self.mu, decimal=6)
        np.testing.assert_almost_equal(self.tvm.kappa, self.kappa, decimal=6)
        self.assertEqual(self.tvm.lambda_, self.lambda_)

    def test_integral(self):
        # test integral
        self.assertAlmostEqual(self.tvm.integrate(), 1, delta=1e-5)

    def test_trigonometric_moment_numerical(self):
        np.testing.assert_almost_equal(
            self.tvm.trigonometric_moment_numerical(0), np.array([1, 1]), decimal=5
        )

    # jscpd:ignore-start
    # pylint: disable=R0801
    def _unnormalized_pdf(self, xs):
        return np.exp(
            self.kappa[0] * np.cos(xs[..., 0] - self.mu[0])
            + self.kappa[1] * np.cos(xs[..., 1] - self.mu[1])
            + self.lambda_ * np.sin(xs[..., 0] - self.mu[0]) * np.sin(xs[..., 1] - self.mu[1])
        )
    # jscpd:ignore-end

    @parameterized.expand(
        [
            (np.array([3, 2]),),
            (np.array([1, 4]),),
            (np.array([5, 6]),),
            (np.array([-3, 11]),),
            (np.array([[5, 1], [6, 3]]),),
            (np.column_stack((np.arange(0, 2 * np.pi, 0.1), np.arange(1 * np.pi, 3 * np.pi, 0.1))),),
        ]
    )
    def test_pdf(self, x):
        C = self.tvm.C

        def pdf(x):
            return self._unnormalized_pdf(x) * C

        expected = pdf(x)

        np.testing.assert_almost_equal(self.tvm.pdf(x), expected, decimal=10)


if __name__ == "__main__":
    unittest.main()
