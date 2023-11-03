import unittest

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import allclose, mean, ones, random, std, zeros
from pyrecest.sampling.euclidean_sampler import GaussianSampler
import numpy.testing as npt

class TestGaussianSampler(unittest.TestCase):
    def setUp(self):
        random.seed(0)
        self.sampler = GaussianSampler()
        self.n_samples = 200
        self.dim = 2
        self.samples = self.sampler.sample_stochastic(self.n_samples, self.dim)

    def test_sample_stochastic(self):
        # Check that the returned shape matches the requested number of samples and dimension
        self.assertEqual(self.samples.shape, (self.n_samples, self.dim))

    def test_gaussian_properties(self):
        # Check that the mean is close to 0 for each dimension
        means = mean(self.samples, axis=0)
        npt.assert_allclose(means, zeros(self.dim), atol=0.15)

        # Check that the standard deviation is close to 1 for each dimension
        std_devs = std(self.samples, axis=0)
        self.assertTrue(allclose(std_devs, ones(self.dim), atol=0.1))


if __name__ == "__main__":
    unittest.main()
