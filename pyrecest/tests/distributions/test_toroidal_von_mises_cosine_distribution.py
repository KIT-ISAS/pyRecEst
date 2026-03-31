import unittest

import matplotlib
import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend
from parameterized import parameterized

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import arange, array, column_stack, cos, exp, pi
from pyrecest.distributions.hypertorus.toroidal_von_mises_cosine_distribution import (
    ToroidalVonMisesCosineDistribution,
)

matplotlib.pyplot.close("all")
matplotlib.use("Agg")


class ToroidalVMCosineDistributionTest(unittest.TestCase):
    def setUp(self):
        self.mu = array([1.0, 2.0])
        self.kappa = array([0.7, 1.4])
        self.kappa3 = array(0.5)
        self.tvm = ToroidalVonMisesCosineDistribution(
            self.mu, self.kappa, self.kappa3
        )

    def test_instance(self):
        self.assertIsInstance(self.tvm, ToroidalVonMisesCosineDistribution)

    def test_mu_kappa_kappa3(self):
        npt.assert_allclose(self.tvm.mu, self.mu)
        npt.assert_allclose(self.tvm.kappa, self.kappa)
        self.assertEqual(self.tvm.kappa3, self.kappa3)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_integral(self):
        self.assertAlmostEqual(self.tvm.integrate(), 1.0, delta=1e-5)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_trigonometric_moment_numerical(self):
        npt.assert_allclose(
            self.tvm.trigonometric_moment_numerical(0), array([1.0, 1.0])
        )

    def test_plot_2d(self):
        self.tvm.plot()

    # jscpd:ignore-start
    # pylint: disable=R0801
    def _unnormalized_pdf(self, xs):
        return exp(
            self.kappa[0] * cos(xs[..., 0] - self.mu[0])
            + self.kappa[1] * cos(xs[..., 1] - self.mu[1])
            - self.kappa3
            * cos(xs[..., 0] - self.mu[0] - xs[..., 1] + self.mu[1])
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

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_trigonometric_moment_analytical(self):
        m_analytical = self.tvm.trigonometric_moment(1)
        m_numerical = self.tvm.trigonometric_moment_numerical(1)
        npt.assert_allclose(m_analytical, m_numerical, rtol=1e-8)

    def test_shift(self):
        shift_by = array([4.0, 2.0])
        tvm2 = self.tvm.shift(shift_by)
        self.assertIsInstance(tvm2, ToroidalVonMisesCosineDistribution)
        x_test = column_stack(
            (arange(0.0, 2.0 * pi, 0.3), arange(0.0, 2.0 * pi, 0.3))
        )
        npt.assert_allclose(
            tvm2.pdf(x_test),
            self.tvm.pdf(x_test - shift_by),
            atol=1e-10,
        )


if __name__ == "__main__":
    unittest.main()
