from pyrecest.backend import linspace
from pyrecest.backend import array
from pyrecest.backend import allclose
from pyrecest.backend import all
import unittest

import numpy as np
import scipy
from pyrecest.distributions import GaussianDistribution
from scipy.stats import multivariate_normal


class GaussianDistributionTest(unittest.TestCase):
    def test_gaussian_distribution_3d(self):
        mu = array([2, 3, 4])
        C = array([[1.1, 0.4, 0], [0.4, 0.9, 0], [0, 0, 0.1]])
        g = GaussianDistribution(mu, C)

        xs = array(
            [
                [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
                [-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7],
                [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8],
            ]
        ).T
        self.assertTrue(
            allclose(g.pdf(xs), multivariate_normal.pdf(xs, mu, C), rtol=1e-10)
        )

        n = 10
        s = g.sample(n)
        self.assertEqual(
            s.shape,
            (
                n,
                3,
            ),
        )

    def test_mode(self):
        mu = array([1, 2, 3])
        C = array([[1.1, 0.4, 0], [0.4, 0.9, 0], [0, 0, 1]])
        g = GaussianDistribution(mu, C)

        self.assertTrue(allclose(g.mode(), mu, atol=1e-6))

    def test_shift(self):
        mu = array([3, 2, 1])
        C = array([[1.1, -0.4, 0], [-0.4, 0.9, 0], [0, 0, 1]])
        g = GaussianDistribution(mu, C)

        shift_by = array([2, -2, 3])
        g_shifted = g.shift(shift_by)

        self.assertTrue(allclose(g_shifted.mode(), mu + shift_by, atol=1e-6))

    def test_marginalization(self):
        mu = array([1, 2])
        C = array([[1.1, 0.4], [0.4, 0.9]])
        g = GaussianDistribution(mu, C)

        grid = linspace(-10, 10, 30)
        dist_marginalized = g.marginalize_out(1)

        def marginlized_1D_via_integrate(xs):
            def integrand(y, x):
                return g.pdf(array([x, y]))

            result = []
            for x_curr in xs:
                integral_value, _ = scipy.integrate.quad(
                    integrand, -np.inf, np.inf, args=x_curr
                )
                result.append(integral_value)
            return array(result)

        self.assertTrue(
            allclose(
                dist_marginalized.pdf(grid),
                marginlized_1D_via_integrate(grid),
                atol=1e-9,
            )
        )


if __name__ == "__main__":
    unittest.main()