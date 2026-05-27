import unittest

import numpy as np
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

    def test_mean_and_covariance(self):
        center = array([2.0, 3.0])
        shape_matrix = array([[4.0, 3.0], [3.0, 9.0]])
        dist = EllipsoidalBallUniformDistribution(center, shape_matrix)

        npt.assert_allclose(dist.mean(), center)
        npt.assert_allclose(dist.covariance(), shape_matrix / (dist.dim + 2))

    def test_sampling(self):
        dist = EllipsoidalBallUniformDistribution(
            array([2.0, 3.0]), array([[4.0, 3.0], [3.0, 9.0]])
        )
        samples = dist.sample(10)
        self.assertEqual(samples.shape[-1], dist.dim)
        self.assertEqual(samples.shape[0], 10.0)
        p = dist.pdf(samples)
        self.assertTrue(all(p == p[0]))

    def test_sampling_accepts_integer_like_count(self):
        dist = EllipsoidalBallUniformDistribution(
            array([2.0, 3.0]), array([[4.0, 3.0], [3.0, 9.0]])
        )

        samples = dist.sample(np.int64(4))

        self.assertEqual(samples.shape, (4, dist.dim))

    def test_sampling_rejects_invalid_count(self):
        dist = EllipsoidalBallUniformDistribution(
            array([2.0, 3.0]), array([[4.0, 3.0], [3.0, 9.0]])
        )

        for n in (0, -1, 1.5, True):
            with self.subTest(n=n):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    dist.sample(n)


if __name__ == "__main__":
    unittest.main()
