import unittest

import matplotlib
import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend
from parameterized import parameterized

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import arange, array, column_stack, cos, exp, pi, sin
from pyrecest.distributions.hypertorus.toroidal_vm_rivest_distribution import (
    ToroidalVMRivestDistribution,
)

matplotlib.pyplot.close("all")
matplotlib.use("Agg")


class ToroidalVMRivestDistributionTest(unittest.TestCase):
    def setUp(self):
        self.mu = array([1.0, 2.0])
        self.kappa = array([0.7, 1.4])
        self.alpha = array(0.5)
        self.beta = array(0.3)
        self.dist = ToroidalVMRivestDistribution(
            self.mu, self.kappa, self.alpha, self.beta
        )

    def test_instance(self):
        self.assertIsInstance(self.dist, ToroidalVMRivestDistribution)

    def test_mu_kappa_alpha_beta(self):
        npt.assert_allclose(self.dist.mu, self.mu)
        npt.assert_allclose(self.dist.kappa, self.kappa)
        self.assertEqual(self.dist.alpha, self.alpha)
        self.assertEqual(self.dist.beta, self.beta)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_integral(self):
        self.assertAlmostEqual(self.dist.integrate(), 1.0, delta=1e-5)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_trigonometric_moment_numerical(self):
        npt.assert_allclose(
            self.dist.trigonometric_moment_numerical(0), array([1.0, 1.0])
        )

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_trigonometric_moment_n1_vs_numerical(self):
        m_analytical = self.dist.trigonometric_moment(1)
        m_numerical = self.dist.trigonometric_moment_numerical(1)
        npt.assert_allclose(m_analytical.real, m_numerical.real, atol=1e-4)
        npt.assert_allclose(m_analytical.imag, m_numerical.imag, atol=1e-4)

    def test_plot_2d(self):
        self.dist.plot()

    def test_shift(self):
        shift = array([0.5, 1.0])
        shifted = self.dist.shift(shift)
        self.assertIsInstance(shifted, ToroidalVMRivestDistribution)
        npt.assert_allclose(shifted.kappa, self.dist.kappa)
        self.assertEqual(shifted.alpha, self.dist.alpha)
        self.assertEqual(shifted.beta, self.dist.beta)

    # jscpd:ignore-start
    # pylint: disable=R0801
    def _unnormalized_pdf(self, xs):
        return exp(
            self.kappa[0] * cos(xs[..., 0] - self.mu[0])
            + self.kappa[1] * cos(xs[..., 1] - self.mu[1])
            + self.alpha * cos(xs[..., 0] - self.mu[0]) * cos(xs[..., 1] - self.mu[1])
            + self.beta * sin(xs[..., 0] - self.mu[0]) * sin(xs[..., 1] - self.mu[1])
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
        C = self.dist.C

        def pdf(x):
            return self._unnormalized_pdf(x) * C

        expected = pdf(x)
        npt.assert_allclose(self.dist.pdf(x), expected)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_circular_correlation_jammalamadaka(self):
        rhoc = self.dist.circular_correlation_jammalamadaka()
        self.assertGreater(rhoc, -1.0)
        self.assertLess(rhoc, 1.0)

    def test_zero_correlation(self):
        # When alpha=beta=0, the distribution factorises into two independent von Mises
        dist_indep = ToroidalVMRivestDistribution(
            self.mu, self.kappa, array(0.0), array(0.0)
        )
        # normalization constant should equal 4*pi^2 * I_0(kappa[0]) * I_0(kappa[1])
        from scipy.special import iv

        expected_cinv = float(
            4.0 * pi**2 * iv(0, float(self.kappa[0])) * iv(0, float(self.kappa[1]))
        )
        self.assertAlmostEqual(1.0 / dist_indep.C, expected_cinv, places=5)


if __name__ == "__main__":
    unittest.main()
