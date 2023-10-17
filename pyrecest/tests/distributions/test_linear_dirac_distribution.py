from pyrecest.backend import eye
from pyrecest.backend import array
from pyrecest.backend import allclose
from pyrecest.backend import all
import unittest

import numpy as np
from pyrecest.distributions import GaussianDistribution
from pyrecest.distributions.nonperiodic.linear_dirac_distribution import (
    LinearDiracDistribution,
)
from scipy.stats import wishart


class LinearDiracDistributionTest(unittest.TestCase):
    def test_from_distribution(self):
        np.random.seed(0)
        C = wishart.rvs(3, eye(3))
        hwn = GaussianDistribution(array([1, 2, 3]), C)
        hwd = LinearDiracDistribution.from_distribution(hwn, 100000)
        self.assertTrue(allclose(hwd.mean(), hwn.mean(), atol=0.005))
        self.assertTrue(allclose(hwd.covariance(), hwn.covariance(), rtol=0.01))

    def test_mean_and_cov(self):
        np.random.seed(0)
        gd = GaussianDistribution(array([1, 2]), array([[2, -0.3], [-0.3, 1]]))
        ddist = LinearDiracDistribution(gd.sample(10000))
        self.assertTrue(allclose(ddist.mean(), gd.mean(), atol=0.05))
        self.assertTrue(allclose(ddist.covariance(), gd.covariance(), atol=0.05))


if __name__ == "__main__":
    unittest.main()