import importlib.util
import unittest

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import allclose, linalg, prod, random

healpy_installed = importlib.util.find_spec("healpy") is not None


from parameterized import parameterized
from pyrecest.sampling.hyperspherical_sampler import get_grid_hypersphere

from ..sampling.hyperspherical_sampler import (
    AbstractHopfBasedS3Sampler,
    DriscollHealySampler,
    FibonacciHopfSampler,
    HealpixHopfSampler,
    HealpixSampler,
    SphericalFibonacciSampler,
)


class TestHypersphericalGridGenerationFunction(unittest.TestCase):
    @parameterized.expand(
        [
            ("healpix", 2, 48, "n_side"),
            ("driscoll_healy", 2, 91, "l_max"),
            ("fibonacci", 12, 12, "n_samples"),
            ("spherical_fibonacci", 12, 12, "n_samples"),
        ]
    )
    @unittest.skipIf(not healpy_installed, "healpy is not installed")
    def test_get_grid_sphere(
        self, method, grid_density_parameter, grid_points_expected, desc_key
    ):
        samples, grid_specific_description = get_grid_hypersphere(
            method, grid_density_parameter
        )

        self.assertEqual(
            samples.shape[0],
            grid_points_expected,
            f"Expected {grid_points_expected} points but got {samples.shape[0]}",
        )
        self.assertEqual(
            samples.shape[1],
            3,
            f"Expected 3-dimensional-output but got {samples.shape[1]}-dimensional output",
        )
        self.assertEqual(grid_specific_description[desc_key], grid_density_parameter)

    @unittest.skipIf(not healpy_installed, "healpy is not installed")
    def test_get_grid_hypersphere(self):
        samples, _ = get_grid_hypersphere("healpix_hopf", 0)

        self.assertEqual(
            samples.shape[0], 72, f"Expected {72} points but got {samples.shape[0]}"
        )
        self.assertEqual(
            samples.shape[1],
            4,
            f"Expected 4-dimensional-output but got {samples.shape[1]}-dimensional output",
        )


class TestHypersphericalSampler(unittest.TestCase):
    @parameterized.expand(
        [
            (HealpixSampler(), 2, 48, "n_side"),
            (DriscollHealySampler(), 2, 91, "l_max"),
            (SphericalFibonacciSampler(), 12, 12, "n_samples"),
        ]
    )
    @unittest.skipIf(not healpy_installed, "healpy is not installed")
    def test_samplers(
        self, sampler, grid_density_parameter, grid_points_expected, desc_key
    ):
        grid, grid_description = sampler.get_grid(grid_density_parameter)

        self.assertEqual(
            grid.shape[0],
            grid_points_expected,
            f"Expected {grid_points_expected} points but got {grid.shape[0]}",
        )
        self.assertEqual(
            grid.shape[1],
            3,
            f"Expected 3-dimensional-output but got {grid.shape[1]}-dimensional output",
        )
        self.assertEqual(grid_description[desc_key], grid_density_parameter)

    @parameterized.expand([(0, 72), (1, 648)])
    @unittest.skipIf(not healpy_installed, "healpy is not installed")
    def test_healpix_hopf_sampler(self, input_value, expected_grid_points):
        sampler = HealpixHopfSampler()
        dim = 3
        grid, _ = sampler.get_grid(input_value)

        self.assertEqual(
            grid.shape[0],
            expected_grid_points,
            f"Expected {expected_grid_points} points but got {grid.shape[1]}",
        )
        self.assertEqual(
            grid.shape[1],
            dim + 1,
            f"Expected {dim+1}-dimensional-output but got {grid.shape[1]}-dimensional output",
        )

    def test_fibonacci_hopf_sampler(self):
        sampler = FibonacciHopfSampler()
        grid_density_parameter = [12, 4]
        grid, _ = sampler.get_grid(grid_density_parameter)

        expected_points = prod(grid_density_parameter)
        self.assertEqual(
            grid.shape[0],
            expected_points,
            f"Expected {expected_points} points but got {grid.shape[0]}",
        )
        self.assertEqual(
            grid.shape[1],
            4,
            f"Expected 4-dimensional-output but got {grid.shape[1]}-dimensional output",
        )


class TestHopfConversion(unittest.TestCase):
    def test_conversion(self):
        # Generate a sample matrix of size (n, 4) containing unit vectors.
        n = 100  # sample size
        random_vectors = random.normal(0.0, 1.0, (n, 4))
        unit_vectors = (
            random_vectors / linalg.norm(random_vectors, axis=1)[:, None]
        )

        # Pass the quaternions through the conversion functions
        θ, ϕ, ψ = AbstractHopfBasedS3Sampler.quaternion_to_hopf_yershova(unit_vectors)
        recovered_quaternions = (
            AbstractHopfBasedS3Sampler.hopf_coordinates_to_quaterion_yershova(θ, ϕ, ψ)
        )

        # Check if the original quaternions are close to the recovered quaternions.
        self.assertTrue(allclose(unit_vectors, recovered_quaternions, atol=1e-8))


if __name__ == "__main__":
    unittest.main()
