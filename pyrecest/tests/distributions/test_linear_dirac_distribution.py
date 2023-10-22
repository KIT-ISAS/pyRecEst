from pyrecest.backend import random
from pyrecest.backend import eye
from pyrecest.backend import array
from pyrecest.backend import allclose
from pyrecest.backend import all
import unittest


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
        hwd = LinearDiracDistribution.from_distribution(hwn, 150000)
        self.assertTrue(allclose(hwd.mean(), hwn.mean(), atol=0.005))
        self.assertTrue(allclose(hwd.covariance(), hwn.covariance(), rtol=0.01))

    def test_mean_and_cov(self):
        random.seed(0)
        gd = GaussianDistribution(array([1.0, 2.0]), array([[2.0, -0.3], [-0.3, 1.0]]))
        ddist = LinearDiracDistribution(gd.sample(10000))
        self.assertTrue(allclose(ddist.mean(), gd.mean(), atol=0.05))
        self.assertTrue(allclose(ddist.covariance(), gd.covariance(), atol=0.05))


if __name__ == "__main__":
    unittest.main()