import unittest

import numpy as np
from pyrecest.sampling.euclidean_sampler import GaussianSampler


class TestGaussianSampler(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.sampler = GaussianSampler()
        self.n_samples = 200
        self.dim = 2
        self.samples = self.sampler.sample_stochastic(self.n_samples, self.dim)

    def test_sample_stochastic(self):
        # Check that the returned shape matches the requested number of samples and dimension
        self.assertEqual(self.samples.shape, (self.n_samples, self.dim))

    def test_gaussian_properties(self):
        # Check that the mean is close to 0 for each dimension
        means = np.mean(self.samples, axis=0)
        self.assertTrue(np.allclose(means, np.zeros(self.dim), atol=0.1))

        # Check that the standard deviation is close to 1 for each dimension
        std_devs = np.std(self.samples, axis=0)
        self.assertTrue(np.allclose(std_devs, np.ones(self.dim), atol=0.1))


if __name__ == "__main__":
    unittest.main()