import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import allclose, array, eye, random, diag
# pylint: disable=no-name-in-module,no-member
import pyrecest.backend
from pyrecest.distributions import GaussianDistribution
from pyrecest.distributions.nonperiodic.linear_dirac_distribution import (
    LinearDiracDistribution,
)
from scipy.stats import wishart
import matplotlib
from parameterized import parameterized

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

    @parameterized.expand([
        (
            "1D Plot",
            array([1.0]),  # 1D mean
            array([[1.0]]),  # 1D covariance
            1  # Dimension
        ),
        (
            "2D Plot",
            array([1.0, 2.0]),  # 2D mean
            array([[2.0, -0.3], [-0.3, 1.0]]),  # 2D covariance
            2  # Dimension
        ),
        (
            "3D Plot",
            array([1.0, 2.0, 3.0]),  # 3D mean
            diag(array([2.0, 1.0, 0.5])),  # 3D covariance (diagonal matrix)
            3  # Dimension
        ),
    ])
    @unittest.skipIf(
        pyrecest.backend.__name__ == "pyrecest.jax",
        reason="Not supported on this backend",
    )
    def test_plot(self, name, mean, cov, dim):
        matplotlib.use("Agg")
        matplotlib.pyplot.close("all")
        
        # Seed the random number generator for reproducibility
        random.seed(0)
        
        # Create GaussianDistribution instance
        gd = GaussianDistribution(mean, cov)
        
        # Sample data and create LinearDiracDistribution instance
        ddist = LinearDiracDistribution(gd.sample(10))
        
        try:
            # Attempt to plot
            ddist.plot()
        except (ValueError, RuntimeError) as e:
            self.fail(f"{name}: Plotting failed for dimension {dim} with error: {e}")

if __name__ == "__main__":
    unittest.main()
