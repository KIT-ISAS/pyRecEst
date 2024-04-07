import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import allclose, array, eye, random
from pyrecest.distributions import GaussianDistribution
from pyrecest.distributions.nonperiodic.linear_dirac_distribution import (
    LinearDiracDistribution,
)
from scipy.stats import wishart


class LinearDiracDistributionTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
