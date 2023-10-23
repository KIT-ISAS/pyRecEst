import unittest

import numpy.testing as npt
# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, diag
from pyrecest.distributions import EllipsoidalBallUniformDistribution


class TestEllipsoidalBallUniformDistribution(unittest.TestCase):
    def test_pdf(self):
        dist = EllipsoidalBallUniformDistribution(
            array([0.0, 0.0, 0.0]), diag(array([4.0, 9.0, 16.0]))
        )
        npt.assert_allclose(dist.pdf(array([0.0, 0.0, 0.0])), 1 / 100.53096491)

    def test_sampling(self):
        dist = EllipsoidalBallUniformDistribution(
            array([2.0, 3.0]), array([[4.0, 3.0], [3.0, 9.0]])
        )
        samples = dist.sample(10)
        self.assertEqual(samples.shape[-1], dist.dim)
        self.assertEqual(samples.shape[0], 10.0)
        p = dist.pdf(samples)
        self.assertTrue(all(p == p[0]))


if __name__ == "__main__":
    unittest.main()
