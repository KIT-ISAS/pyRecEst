from math import pi
from pyrecest.backend import std
from pyrecest.backend import all
import unittest


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
        self.assertTrue(all(samples >= 0))
        self.assertTrue(all(samples < 2 * pi))

    def test_get_grid(self):
        grid_density_parameter = 100
        grid_points = self.sampler.get_grid(grid_density_parameter)

        # Check that the returned shape matches the grid density parameter
        self.assertEqual(grid_points.shape[0], grid_density_parameter)

        # Check that all grid points are within the range [0, 2*pi)
        self.assertTrue(all(grid_points >= 0))
        self.assertTrue(all(grid_points < 2 * pi))

        # Check that the grid points are equidistant
        diff = np.diff(grid_points)
        self.assertAlmostEqual(std(diff), 0, places=5)


if __name__ == "__main__":
    unittest.main()