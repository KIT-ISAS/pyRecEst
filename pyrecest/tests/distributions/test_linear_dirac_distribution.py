import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend
from parameterized import parameterized

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import allclose, array, diag, eye, random
from pyrecest.distributions import GaussianDistribution
from pyrecest.distributions.nonperiodic.linear_dirac_distribution import (
    LinearDiracDistribution,
)
from scipy.stats import wishart

from .test_abstract_dirac_distribution import TestAbstractDiracDistribution


class LinearDiracDistributionTest(TestAbstractDiracDistribution):
    def test_from_distribution(self):
        random.seed(0)
        C = wishart.rvs(3, eye(3))
        hwn = GaussianDistribution(array([1.0, 2.0, 3.0]), array(C))
        hwd = LinearDiracDistribution.from_distribution(hwn, 200000)
        npt.assert_allclose(hwd.mean(), hwn.mean(), atol=0.015)
        npt.assert_allclose(hwd.covariance(), hwn.covariance(), atol=0.08)

    def test_mean_and_cov(self):
        random.seed(0)
        gd = GaussianDistribution(array([1.0, 2.0]), array([[2.0, -0.3], [-0.3, 1.0]]))
        ddist = LinearDiracDistribution(gd.sample(15000))
        self.assertTrue(allclose(ddist.mean(), gd.mean(), atol=0.05))
        self.assertTrue(allclose(ddist.covariance(), gd.covariance(), atol=0.05))

    @parameterized.expand(
        [
            (
                "1D Plot",
                GaussianDistribution(
                    array([1.0]), array([[1.0]])  # 1D mean
                ),  # 1D covariance
                1,  # Dimension
            ),
            (
                "2D Plot",
                GaussianDistribution(
                    array([1.0, 2.0]), array([[2.0, -0.3], [-0.3, 1.0]])  # 2D mean
                ),  # 2D covariance
                2,  # Dimension
            ),
            (
                "3D Plot",
                GaussianDistribution(
                    array([1.0, 2.0, 3.0]), diag(array([2.0, 1.0, 0.5]))  # 3D mean
                ),  # 3D covariance (diagonal matrix)
                3,  # Dimension
            ),
        ]
    )
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_plot(self, name, dist, dim):
        self._test_plot_helper(name, dist, dim, LinearDiracDistribution)


if __name__ == "__main__":
    unittest.main()
