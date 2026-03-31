import unittest

import numpy.testing as npt
import scipy.linalg

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array
from pyrecest.distributions.cart_prod.partially_wrapped_normal_distribution import (
    PartiallyWrappedNormalDistribution,
)


class TestPartiallyWrappedNormalDistribution(unittest.TestCase):
    def setUp(self) -> None:
        self.mu = array([5.0, 1.0])
        self.C = array([[2.0, 1.0], [1.0, 1.0]])
        self.dist_2d = PartiallyWrappedNormalDistribution(self.mu, self.C, 1)

    def test_pdf(self):
        # Use distinct rows so that a tile/repeat swap in the implementation would
        # mix contributions between different input points and produce wrong values.
        xs = array([[0.5, 1.0], [1.5, 0.5], [2.0, 2.0]])
        result = self.dist_2d.pdf(xs)
        self.assertEqual(result.shape, (3,))
        # Each row in the batch must match its individually evaluated value.
        for i in range(xs.shape[0]):
            npt.assert_allclose(
                result[i],
                self.dist_2d.pdf(xs[i : i + 1])[0],  # noqa: E203
                rtol=1e-10,
            )

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
            self.dist_2d.hybrid_moment(),
            [0.10435348, -0.35276852, self.mu[-1]],
            rtol=5e-7,
        )


if __name__ == "__main__":
    unittest.main()
