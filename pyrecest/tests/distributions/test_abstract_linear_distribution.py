from pyrecest.backend import squeeze
from pyrecest.backend import isclose
from pyrecest.backend import array
import unittest

import matplotlib
import numpy as np
from pyrecest.distributions import (
    AbstractLinearDistribution,
    CustomLinearDistribution,
    GaussianDistribution,
)

matplotlib.use("Agg")


class TestAbstractLinearDistribution(unittest.TestCase):
    def setUp(self):
        self.mu_2D = array([5, 1])
        self.C_2D = array([[2, 1], [1, 1]])
        self.mu_3D = array([1, 2, 3])
        self.C_3D = array([[1.1, 0.4, 0], [0.4, 0.9, 0], [0, 0, 1]])
        self.g_2D = GaussianDistribution(self.mu_2D, self.C_2D)
        self.g_3D = GaussianDistribution(self.mu_3D, self.C_3D)

    def test_integrate_numerically(self):
        """Test that the numerical integration of a Gaussian distribution equals 1."""
        dist = GaussianDistribution(array([1, 2]), np.diag([1, 2]))
        integration_result = dist.integrate_numerically()
        assert isclose(
            integration_result, 1, rtol=1e-5
        ), f"Expected 1, but got {integration_result}"

    def test_integrate_fun_over_domain(self):
        dist = GaussianDistribution(array([1, 2]), np.diag([1, 2]))

        def f(x):
            return 0.3 * dist.pdf(x)

        dim = 2
        left = [-np.inf, -np.inf]
        right = [np.inf, np.inf]

        integration_result = AbstractLinearDistribution.integrate_fun_over_domain(
            f, dim, left, right
        )
        assert isclose(
            integration_result, 0.3, rtol=1e-5
        ), f"Expected 0.3, but got {integration_result}"

    def test_mode_numerical_custom_1D(self):
        cd = CustomLinearDistribution(
            lambda x: squeeze(
                ((x > -1) & (x <= 0)) * (1 + x) + ((x > 0) & (x <= 1)) * (1 - x)
            ),
            1,
        )
        cd = cd.shift(array(0.5))
        self.assertAlmostEqual(cd.mode_numerical(), 0.5, delta=1e-4)

    def test_mean_numerical_gaussian_2D(self):
        np.testing.assert_allclose(self.g_2D.mean_numerical(), self.mu_2D, atol=1e-6)

    def test_mode_numerical_gaussian_2D_mean_far_away(self):
        mu = array([5, 10])
        C = array([[2, 1], [1, 1]])
        g = GaussianDistribution(mu, C)
        np.testing.assert_allclose(g.mode_numerical(), mu, atol=2e-4)

    def test_mode_numerical_gaussian_3D(self):
        np.testing.assert_allclose(self.g_3D.mode_numerical(), self.mu_3D, atol=5e-4)

    def test_covariance_numerical_gaussian_2D(self):
        np.testing.assert_allclose(
            self.g_2D.covariance_numerical(), self.C_2D, atol=1e-6
        )

    def test_plot_state_r2(self):
        gd = GaussianDistribution(array([1, 2]), array([[1, 0.5], [0.5, 1]]))
        gd.plot()


if __name__ == "__main__":
    unittest.main()