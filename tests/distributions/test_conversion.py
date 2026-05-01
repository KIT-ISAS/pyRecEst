import unittest

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import allclose, array, eye, pi, random
from pyrecest.distributions import (
    CircularDiracDistribution,
    CircularFourierDistribution,
    CircularGridDistribution,
    GaussianDistribution,
    LinearDiracDistribution,
    VonMisesDistribution,
)
from pyrecest.distributions.conversion import (
    ConversionError,
    ConversionResult,
    can_convert,
    convert_distribution,
    register_conversion_alias,
)


class ConversionTest(unittest.TestCase):
    def test_convert_distribution_uses_target_from_distribution(self):
        random.seed(0)
        gaussian = GaussianDistribution(array([0.0, 0.0]), eye(2))

        particles = convert_distribution(
            gaussian, LinearDiracDistribution, n_particles=25
        )

        self.assertIsInstance(particles, LinearDiracDistribution)
        self.assertEqual(particles.d.shape[0], 25)

    def test_return_info(self):
        random.seed(0)
        gaussian = GaussianDistribution(array([0.0, 0.0]), eye(2))

        result = convert_distribution(
            gaussian,
            LinearDiracDistribution,
            n_particles=5,
            return_info=True,
        )

        self.assertIsInstance(result, ConversionResult)
        self.assertIsInstance(result.distribution, LinearDiracDistribution)
        self.assertEqual(result.source_type, GaussianDistribution)
        self.assertEqual(result.target_type, LinearDiracDistribution)
        self.assertEqual(result.method, "LinearDiracDistribution.from_distribution")
        self.assertFalse(result.exact)

    def test_identity_conversion_is_exact(self):
        gaussian = GaussianDistribution(array([0.0, 0.0]), eye(2))

        result = convert_distribution(
            gaussian, GaussianDistribution, return_info=True
        )

        self.assertIs(result.distribution, gaussian)
        self.assertTrue(result.exact)
        self.assertEqual(result.method, "identity")

    def test_can_convert_reports_route_only(self):
        gaussian = GaussianDistribution(array([0.0, 0.0]), eye(2))

        self.assertTrue(can_convert(gaussian, LinearDiracDistribution))

    def test_missing_required_conversion_argument_raises_helpful_error(self):
        gaussian = GaussianDistribution(array([0.0, 0.0]), eye(2))

        with self.assertRaises(ConversionError):
            convert_distribution(gaussian, LinearDiracDistribution)

    def test_unknown_conversion_argument_raises_helpful_error(self):
        gaussian = GaussianDistribution(array([0.0, 0.0]), eye(2))

        with self.assertRaises(ConversionError):
            convert_distribution(
                gaussian,
                LinearDiracDistribution,
                n_particles=5,
                wrong_name=True,
            )

    def test_manifold_specific_distribution_supports_approximate_as(self):
        random.seed(0)
        gaussian = GaussianDistribution(array([0.0, 0.0]), eye(2))

        particles = gaussian.approximate_as(
            LinearDiracDistribution, n_particles=25
        )

        self.assertIsInstance(particles, LinearDiracDistribution)
        self.assertEqual(particles.d.shape[0], 25)

    def test_linear_dirac_to_gaussian_uses_moment_matching(self):
        particles = LinearDiracDistribution(
            array([[0.0, 0.0], [2.0, 0.0]]), array([0.5, 0.5])
        )

        gaussian = convert_distribution(particles, GaussianDistribution)

        self.assertIsInstance(gaussian, GaussianDistribution)
        self.assertTrue(allclose(gaussian.mean(), particles.mean()))
        self.assertTrue(allclose(gaussian.covariance(), particles.covariance()))

    def test_builtin_string_alias_particles(self):
        random.seed(0)
        gaussian = GaussianDistribution(array([0.0, 0.0]), eye(2))

        particles = convert_distribution(gaussian, "particles", n_particles=25)

        self.assertIsInstance(particles, LinearDiracDistribution)
        self.assertEqual(particles.d.shape[0], 25)

    def test_builtin_string_alias_gaussian(self):
        particles = LinearDiracDistribution(
            array([[0.0, 0.0], [2.0, 0.0]]), array([0.5, 0.5])
        )

        gaussian = particles.approximate_as("gaussian")

        self.assertIsInstance(gaussian, GaussianDistribution)
        self.assertTrue(allclose(gaussian.mean(), particles.mean()))

    def test_custom_string_alias(self):
        random.seed(0)
        register_conversion_alias("test_particles", LinearDiracDistribution)
        gaussian = GaussianDistribution(array([0.0, 0.0]), eye(2))

        particles = convert_distribution(gaussian, "test_particles", n_particles=5)

        self.assertIsInstance(particles, LinearDiracDistribution)

    def test_unknown_string_alias_raises_helpful_error(self):
        gaussian = GaussianDistribution(array([0.0, 0.0]), eye(2))

        with self.assertRaises(ConversionError):
            convert_distribution(gaussian, "not_a_representation")

    def test_can_convert_supports_string_aliases(self):
        gaussian = GaussianDistribution(array([0.0, 0.0]), eye(2))

        self.assertTrue(can_convert(gaussian, "particles"))
        self.assertFalse(can_convert(gaussian, "not_a_representation"))

    def test_circular_von_mises_to_particles_alias(self):
        random.seed(0)
        distribution = VonMisesDistribution(array(0.4), 2.5)

        particles = distribution.approximate_as("particles", n_particles=20)

        self.assertIsInstance(particles, CircularDiracDistribution)
        self.assertEqual(particles.d.shape[0], 20)

    def test_circular_von_mises_to_grid_alias(self):
        distribution = VonMisesDistribution(array(0.4), 2.5)

        grid = distribution.approximate_as("grid", no_of_gridpoints=32)

        self.assertIsInstance(grid, CircularGridDistribution)
        self.assertEqual(grid.grid_values.shape[0], 32)
        self.assertTrue(allclose(grid.grid_values, distribution.pdf(grid.get_grid())))

    def test_circular_von_mises_to_fourier_alias(self):
        distribution = VonMisesDistribution(array(0.4), 2.5)

        fourier = distribution.approximate_as("fourier", n=32)

        self.assertIsInstance(fourier, CircularFourierDistribution)
        self.assertEqual(fourier.n, 32)
        self.assertTrue(allclose(fourier.integrate(), 1.0, atol=5e-2))

    def test_circular_grid_to_fourier_alias(self):
        grid = VonMisesDistribution(array(0.4), 2.5).approximate_as(
            "grid", no_of_gridpoints=32
        )

        fourier = grid.approximate_as("fourier", n=32)

        self.assertIsInstance(fourier, CircularFourierDistribution)
        self.assertEqual(fourier.n, 32)

    def test_circular_fourier_to_grid_alias(self):
        fourier = VonMisesDistribution(array(0.4), 2.5).approximate_as(
            "fourier", n=32
        )

        grid = fourier.approximate_as("grid", no_of_gridpoints=32)

        self.assertIsInstance(grid, CircularGridDistribution)
        self.assertEqual(grid.grid_values.shape[0], 32)
        self.assertTrue(allclose(grid.grid_values, fourier.pdf(grid.get_grid())))

    def test_circular_dirac_to_fourier_identity(self):
        particles = CircularDiracDistribution(
            array([0.0, pi]), array([0.25, 0.75])
        )

        fourier = particles.approximate_as(
            "fourier",
            n=9,
            transformation="identity",
            store_values_multiplied_by_n=False,
        )

        self.assertIsInstance(fourier, CircularFourierDistribution)
        self.assertEqual(fourier.n, 9)
        self.assertFalse(fourier.multiplied_by_n)
        self.assertTrue(allclose(fourier.get_c()[0], 1.0 / (2.0 * pi)))


if __name__ == "__main__":
    unittest.main()
