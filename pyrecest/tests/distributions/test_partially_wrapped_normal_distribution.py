import unittest

import numpy.testing as npt
import scipy.linalg

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, ones
from pyrecest.distributions.cart_prod.partially_wrapped_normal_distribution import (
    PartiallyWrappedNormalDistribution,
)


class TestPartiallyWrappedNormalDistribution(unittest.TestCase):
    def setUp(self) -> None:
        self.mu = array([5.0, 1.0])
        self.C = array([[2.0, 1.0], [1.0, 1.0]])
        self.dist_2d = PartiallyWrappedNormalDistribution(self.mu, self.C, 1)

    def test_pdf(self):
        self.assertEqual(self.dist_2d.pdf(ones((10, 2))).shape, (10,))

    def test_hybrid_mean_2d(self):
        npt.assert_allclose(self.dist_2d.hybrid_mean(), self.mu)

    def test_hybrid_mean_4d(self):
        mu = array([5.0, 1.0, 3.0, 4.0])
        C = array(
            scipy.linalg.block_diag([[2.0, 1.0], [1.0, 1.0]], [[2.0, 1.0], [1.0, 1.0]])
        )
        dist = PartiallyWrappedNormalDistribution(mu, C, 2)
        npt.assert_allclose(dist.hybrid_mean(), mu)

    def test_hybrid_moment_2d(self):
        # Validate against precalculated values
        npt.assert_allclose(
            self.dist_2d.hybrid_moment(), [0.10435348, -0.35276852, self.mu[-1]], rtol=5e-7
        )


if __name__ == "__main__":
    unittest.main()
