import unittest

import numpy as np
from pyrecest.sampling.hypertoroidal_sampler import CircularUniformSampler


class TestCircularUniformSampler(unittest.TestCase):
    def setUp(self):
        self.sampler = CircularUniformSampler()

    def test_sample_stochastic(self):
        n_samples = 1000
        samples = self.sampler.sample_stochastic(n_samples)

        # Check that the returned shape matches the requested number of samples
        self.assertEqual(samples.shape[0], n_samples)

        # Check that all samples are within the range [0, 2*pi)
        self.assertTrue(np.all(samples >= 0))
        self.assertTrue(np.all(samples < 2 * np.pi))

    def test_get_grid(self):
        grid_density_parameter = 100
        grid_points = self.sampler.get_grid(grid_density_parameter)

        # Check that the returned shape matches the grid density parameter
        self.assertEqual(grid_points.shape[0], grid_density_parameter)

        # Check that all grid points are within the range [0, 2*pi)
        self.assertTrue(np.all(grid_points >= 0))
        self.assertTrue(np.all(grid_points < 2 * np.pi))

        # Check that the grid points are equidistant
        diff = np.diff(grid_points)
        self.assertAlmostEqual(np.std(diff), 0, places=5)


if __name__ == "__main__":
    unittest.main()
