import unittest
import numpy as np
from pyrecest.distributions.hypertorus.toroidal_uniform_distribution import ToroidalUniformDistribution
import warnings

class TestToroidalUniformDistribution(unittest.TestCase):

    def setUp(self):
        self.tud = ToroidalUniformDistribution()
        self.x = np.tile(np.array([[1, 2, 3, 4, 5, 6]]), (2, 1))

    def test_pdf(self):
        self.assertTrue(np.allclose(self.tud.pdf(self.x), (1 / (2 * np.pi) ** 2) * np.ones(self.x.shape[1])))

    def test_shift(self):
        tud_shifted = self.tud.shift(np.array([1, 2]))
        self.assertTrue(np.allclose(tud_shifted.pdf(self.x), (1 / (2 * np.pi) ** 2) * np.ones(self.x.shape[1])))

    def test_trigonometric_moments(self):
        for k in range(4):
            self.assertTrue(np.allclose(self.tud.trigonometric_moment(k), self.tud.trigonometric_moment_numerical(k), atol=1e-10))
            if k == 0:
                self.assertTrue(np.allclose(self.tud.trigonometric_moment(k), np.ones(2), rtol=1e-10))
            else:
                self.assertTrue(np.allclose(self.tud.trigonometric_moment(k), np.zeros(2), rtol=1e-10))

    def test_circular_mean(self):
        with warnings.catch_warnings(record=True) as w:
            self.tud.circular_mean()
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[-1].category, RuntimeWarning))
            self.assertIn('Circular uniform distribution does not have a unique mean', str(w[-1].message))

    def test_entropy(self):
        self.assertAlmostEqual(self.tud.entropy(), self.tud.entropy_numerical(), delta=1e-10)

    def test_sampling(self):
        n = 10
        s = self.tud.sample(n)
        self.assertEqual(s.shape, (2, n))
        self.assertTrue(np.all(s >= 0))
        self.assertTrue(np.all(s < 2 * np.pi))


if __name__ == "__main__":
    unittest.main()