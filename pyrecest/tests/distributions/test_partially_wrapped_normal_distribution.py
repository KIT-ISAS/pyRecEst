import unittest

import numpy as np
import scipy.linalg
from pyrecest.distributions.cart_prod.partially_wrapped_normal_distribution import (
    PartiallyWrappedNormalDistribution,
)


class TestPartiallyWrappedNormalDistribution(unittest.TestCase):
    def test_pdf(self):
        mu = np.array([5, 1])
        C = np.array([[2, 1], [1, 1]])
        dist = PartiallyWrappedNormalDistribution(mu, C, 1)
        self.assertEqual(dist.pdf(np.ones((10, 2))).shape, (10,))

    def test_hybrid_mean_2d(self):
        mu = np.array([5, 1])
        C = np.array([[2, 1], [1, 1]])
        dist = PartiallyWrappedNormalDistribution(mu, C, 1)
        np.testing.assert_allclose(dist.hybrid_mean(), mu)

    def test_hybrid_mean_4d(self):
        mu = np.array([5, 1, 3, 4])
        C = np.array(scipy.linalg.block_diag([[2, 1], [1, 1]], [[2, 1], [1, 1]]))
        dist = PartiallyWrappedNormalDistribution(mu, C, 2)
        np.testing.assert_allclose(dist.hybrid_mean(), mu)


if __name__ == "__main__":
    unittest.main()
