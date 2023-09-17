import numpy as np
import unittest
from pyrecest.sampling.euclidean_sampler import AbstractGaussianSampler

class TestAbstractGaussianSampler(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        self.sampler = AbstractGaussianSampler()
        self.n_samples = 100
        self.dim_stochastic = 3
        self.dim_gaussian = 2
        self.samples_stochastic = self.sampler.sample_stochastic(self.n_samples, self.dim_stochastic)
        self.samples_gaussian = self.sampler.sample_stochastic(self.n_samples, self.dim_gaussian)

    def test_sample_stochastic(self):
        # Check that the returned shape matches the requested number of samples and dimension
        self.assertEqual(self.samples_stochastic.shape, (self.n_samples, self.dim_stochastic))

    def test_gaussian_properties(self):
        # Check that the mean is close to 0 for each dimension
        means = np.mean(self.samples_gaussian, axis=0)
        self.assertTrue(np.allclose(means, np.zeros(self.dim_gaussian), atol=0.1))

        # Check that the standard deviation is close to 1 for each dimension
        std_devs = np.std(self.samples_gaussian, axis=0)
        self.assertTrue(np.allclose(std_devs, np.ones(self.dim_gaussian), atol=0.1))

if __name__ == "__main__":
    unittest.main()