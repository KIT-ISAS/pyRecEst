import unittest
from math import pi

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend
from parameterized import parameterized

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import arange, array, column_stack, cos, exp, sin
from pyrecest.distributions.hypertorus.toroidal_von_mises_sine_distribution import (
    ToroidalVonMisesSineDistribution,
)


class ToroidalVMSineDistributionTest(unittest.TestCase):
    def setUp(self):
        self.mu = array([1.0, 2.0])
        self.kappa = array([0.7, 1.4])
        self.lambda_ = array(0.5)
        self.tvm = ToroidalVonMisesSineDistribution(self.mu, self.kappa, self.lambda_)

    def test_instance(self):
        # sanity check
        self.assertIsInstance(self.tvm, ToroidalVonMisesSineDistribution)

    def test_mu_kappa_lambda(self):
        npt.assert_allclose(self.tvm.mu, self.mu)
        npt.assert_allclose(self.tvm.kappa, self.kappa)
        self.assertEqual(self.tvm.lambda_, self.lambda_)

    @unittest.skipIf(
        pyrecest.backend.__name__ == "pyrecest.pytorch",
        reason="Not supported on PyTorch backend",
    )
    def test_integral(self):
        # test integral
        self.assertAlmostEqual(self.tvm.integrate(), 1.0, delta=1e-5)

    def test_trigonometric_moment_numerical(self):
        npt.assert_allclose(
            self.tvm.trigonometric_moment_numerical(0), array([1.0, 1.0])
        )

    # jscpd:ignore-start
    # pylint: disable=R0801
    def _unnormalized_pdf(self, xs):
        return exp(
            self.kappa[0] * cos(xs[..., 0] - self.mu[0])
            + self.kappa[1] * cos(xs[..., 1] - self.mu[1])
            + self.lambda_ * sin(xs[..., 0] - self.mu[0]) * sin(xs[..., 1] - self.mu[1])
        )

    # jscpd:ignore-end

    @parameterized.expand(
        [
            (array([3.0, 2.0]),),
            (array([1.0, 4.0]),),
            (array([5.0, 6.0]),),
            (array([-3.0, 11.0]),),
            (array([[5.0, 1.0], [6.0, 3.0]]),),
            (
                column_stack(
                    (arange(0.0, 2.0 * pi, 0.1), arange(1.0 * pi, 3.0 * pi, 0.1))
                ),
            ),
        ]
    )
    def test_pdf(self, x):
        C = self.tvm.C

        def pdf(x):
            return self._unnormalized_pdf(x) * C

        expected = pdf(x)

        npt.assert_allclose(self.tvm.pdf(x), expected)


if __name__ == "__main__":
    unittest.main()
