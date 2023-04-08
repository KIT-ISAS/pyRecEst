import unittest
import numpy as np
from pyrecest.distributions import EllipsoidalBallUniformDistribution


class TestEllipsoidalBallUniformDistribution(unittest.TestCase):

    def test_pdf(self):
        dist = EllipsoidalBallUniformDistribution(
            np.array([0, 0, 0]), np.diag([4, 9, 16]))
        self.assertAlmostEqual(dist.pdf(np.array([0, 0, 0])), 1/100.53096491)

    def test_sampling(self):
        dist = EllipsoidalBallUniformDistribution(
            np.array([2, 3]), np.array([[4, 3], [3, 9]]))
        samples = dist.sample(10)
        self.assertEqual(samples.shape[-1], dist.dim)
        self.assertEqual(samples.shape[0], 10)
        p = dist.pdf(samples)
        self.assertTrue(all(p == p[0]))

if __name__ == '__main__':
    unittest.main()
