import unittest

import numpy as np
import scipy.linalg
from pyrecest.distributions.cart_prod.partially_wrapped_normal_distribution import (
    PartiallyWrappedNormalDistribution,
)


class TestPartiallyWrappedNormalDistribution(unittest.TestCase): 
    def setUp(self) -> None:
        self.mu = np.array([5, 1])
        self.C = np.array([[2, 1], [1, 1]])
        self.dist_2d = PartiallyWrappedNormalDistribution(self.mu, self.C, 1)
        
    def test_pdf(self):
        self.assertEqual(self.dist_2d.pdf(np.ones((10, 2))).shape, (10,))

    def test_hybrid_mean_2d(self):
        np.testing.assert_allclose(self.dist_2d.hybrid_mean(), self.mu)

    def test_hybrid_mean_4d(self):
        mu = np.array([5, 1, 3, 4])
        C = np.array(scipy.linalg.block_diag([[2, 1], [1, 1]], [[2, 1], [1, 1]]))
        dist = PartiallyWrappedNormalDistribution(mu, C, 2)
        np.testing.assert_allclose(dist.hybrid_mean(), mu)
        
    def test_hybrid_moment_2d(self):
        # Validate against precalculated values
        np.testing.assert_allclose(self.dist_2d.hybrid_moment(), [0.10435348, -0.35276852, self.mu[-1]])


if __name__ == "__main__":
    unittest.main()
