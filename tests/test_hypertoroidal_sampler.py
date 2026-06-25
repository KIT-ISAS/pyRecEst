import unittest

import numpy as np
import numpy.testing as npt

# pylint: disable=redefined-builtin,no-name-in-module,no-member
# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import all, diff, pi, std
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
        self.assertTrue(all(samples >= 0.0))
        self.assertTrue(all(samples < 2.0 * pi))

    def test_sample_stochastic_accepts_integer_like_scalars(self):
        samples = self.sampler.sample_stochastic(np.array(3.0), dim=np.array(1.0))

        self.assertEqual(samples.shape, (3, 1))

    def test_sample_stochastic_rejects_invalid_controls(self):
        invalid_sample_counts = (-1, 1.5, True, [2], "3", b"3")
        for n_samples in invalid_sample_counts:
            with self.subTest(n_samples=n_samples):
                with self.assertRaises(ValueError):
                    self.sampler.sample_stochastic(n_samples)

        invalid_dims = (0, 1.5, 2, True, [1], "1", b"1")
        for dim in invalid_dims:
            with self.subTest(dim=dim):
                with self.assertRaises(ValueError):
                    self.sampler.sample_stochastic(2, dim=dim)

    def test_get_grid(self):
        grid_density_parameter = 100
        grid_points = self.sampler.get_grid(grid_density_parameter)

        # Check that the returned shape matches the grid density parameter
        self.assertEqual(grid_points.shape[0], grid_density_parameter)

        # Check that all grid points are within the range [0, 2*pi)
        self.assertTrue(all(grid_points >= 0.0))
        self.assertTrue(all(grid_points < 2.0 * pi))

        # Check that the grid points are equidistant
        npt.assert_array_almost_equal(std(diff(grid_points)), 0.0)

    def test_get_grid_rejects_nonpositive_density(self):
        with self.assertRaises(ValueError):
            self.sampler.get_grid(0)
        with self.assertRaises(ValueError):
            self.sampler.get_grid(-1)

    def test_get_grid_accepts_integer_like_scalar_density(self):
        grid_points = self.sampler.get_grid(np.array(4.0))

        self.assertEqual(grid_points.shape[0], 4)

    def test_get_grid_rejects_noninteger_density(self):
        for density in (2.5, True, [3], "3", b"3"):
            with self.subTest(density=density):
                with self.assertRaises(ValueError):
                    self.sampler.get_grid(density)


if __name__ == "__main__":
    unittest.main()
