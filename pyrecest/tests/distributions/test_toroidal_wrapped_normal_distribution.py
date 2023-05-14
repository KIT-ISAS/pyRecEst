import unittest
import numpy as np
from pyrecest.distributions.hypertorus.toroidal_wrapped_normal_distribution import ToroidalWrappedNormalDistribution

class TestToroidalWrappedNormalDistribution(unittest.TestCase):

    def setUp(self):
        self.mu = np.array([1, 2])
        self.C = np.array([[1.3, -0.9],
                           [-0.9, 1.2]])
        self.twn = ToroidalWrappedNormalDistribution(self.mu, self.C)

    def test_sanity_check(self):
        self.assertIsInstance(self.twn, ToroidalWrappedNormalDistribution)
        self.assertTrue(np.allclose(self.twn.mu, self.mu))
        self.assertTrue(np.allclose(self.twn.C, self.C))

    def test_integrate(self):
        self.assertAlmostEqual(self.twn.integrate(), 1, delta=1E-5)
        self.assertTrue(np.allclose(self.twn.trigonometric_moment(0), np.array([1, 1]), rtol=1E-5))

    def test_sampling(self):
        n_samples = 5
        s = self.twn.sample(n_samples)
        self.assertEqual(s.shape, (n_samples, 2))
        self.assertTrue(np.allclose(s, np.mod(s, 2 * np.pi)))

if __name__ == '__main__':
    unittest.main()
