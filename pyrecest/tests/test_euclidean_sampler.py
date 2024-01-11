import unittest

import numpy.testing as npt
from pyrecest.backend import mean, ones, random, std, zeros
from pyrecest.sampling.euclidean_sampler import GaussianSampler
from scipy.stats import shapiro


class TestGaussianSampler(unittest.TestCase):
    def setUp(self):
        random.seed(0)
        self.sampler = GaussianSampler()
        self.n_samples = 200
        self.dim = 2
        self.samples = self.sampler.sample_stochastic(self.n_samples, self.dim)

    def test_sample_stochastic_shape(self):
        """Check that the returned shape matches the requested number of samples and dimension."""
        self.assertEqual(self.samples.shape, (self.n_samples, self.dim))

    def test_sample_mean_close_to_zero(self):
        """Check that the mean is close to 0 for each dimension."""
        means = mean(self.samples, axis=0)
        npt.assert_allclose(means, zeros(self.dim), atol=0.15)

    def test_sample_std_dev_close_to_one(self):
        """Check that the standard deviation is close to 1 for each dimension."""
        std_devs = std(self.samples, axis=0)
        npt.assert_allclose(std_devs, ones(self.dim), atol=0.1)

    def test_samples_follow_gaussian_distribution(self):
        """Test if the samples follow a Gaussian distribution for each dimension."""
        for i in range(self.dim):
            _, p_value = shapiro(self.samples[:, i])
            self.assertGreater(p_value, 0.05)


if __name__ == "__main__":
    unittest.main()
