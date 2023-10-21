from pyrecest.backend import column_stack
from math import pi
from pyrecest.backend import sin
from pyrecest.backend import exp
from pyrecest.backend import cos
from pyrecest.backend import array
from pyrecest.backend import arange
import unittest
import numpy.testing as npt


from parameterized import parameterized
from pyrecest.distributions.hypertorus.toroidal_von_mises_sine_distribution import (
    ToroidalVonMisesSineDistribution,
)


class ToroidalVMSineDistributionTest(unittest.TestCase):
    def setUp(self):
        self.mu = array([1, 2])
        self.kappa = array([0.7, 1.4])
        self.lambda_ = array(0.5)
        self.tvm = ToroidalVonMisesSineDistribution(self.mu, self.kappa, self.lambda_)

    def test_instance(self):
        # sanity check
        self.assertIsInstance(self.tvm, ToroidalVonMisesSineDistribution)

    def test_mu_kappa_lambda(self):
        npt.assert_almost_equal(self.tvm.mu, self.mu, decimal=6)
        npt.assert_almost_equal(self.tvm.kappa, self.kappa, decimal=6)
        self.assertEqual(self.tvm.lambda_, self.lambda_)

    def test_integral(self):
        # test integral
        self.assertAlmostEqual(self.tvm.integrate(), 1.0, delta=1e-5)

    def test_trigonometric_moment_numerical(self):
        npt.assert_almost_equal(
            self.tvm.trigonometric_moment_numerical(0), array([1, 1]), decimal=5
        )

    # jscpd:ignore-start
    # pylint: disable=R0801
    def _unnormalized_pdf(self, xs):
        return exp(
            self.kappa[0] * cos(xs[..., 0] - self.mu[0])
            + self.kappa[1] * cos(xs[..., 1] - self.mu[1])
            + self.lambda_
            * sin(xs[..., 0] - self.mu[0])
            * sin(xs[..., 1] - self.mu[1])
        )

    # jscpd:ignore-end

    @parameterized.expand(
        [
            (array([3, 2]),),
            (array([1, 4]),),
            (array([5, 6]),),
            (array([-3, 11]),),
            (array([[5, 1], [6, 3]]),),
            (
                column_stack(
                    (arange(0, 2 * pi, 0.1), arange(1 * pi, 3 * pi, 0.1))
                ),
            ),
        ]
    )
    def test_pdf(self, x):
        C = self.tvm.C

        def pdf(x):
            return self._unnormalized_pdf(x) * C

        expected = pdf(x)

        npt.assert_almost_equal(self.tvm.pdf(x), expected, decimal=10)


if __name__ == "__main__":
    unittest.main()