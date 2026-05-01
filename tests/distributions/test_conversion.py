import unittest

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import allclose, array, eye, random, sum
from pyrecest.distributions import (
    GaussianDistribution,
    HypertoroidalDiracDistribution,
    HypertoroidalGridDistribution,
    HypertoroidalWrappedNormalDistribution,
    LinearDiracDistribution,
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

    def test_hypertoroidal_particles_alias_samples_distribution(self):
        random.seed(0)
        distribution = HypertoroidalWrappedNormalDistribution(
            array([0.1, 1.0]), 0.05 * eye(2)
        )

        particles = convert_distribution(
            distribution, "particles", n_particles=11
        )

        self.assertIsInstance(particles, HypertoroidalDiracDistribution)
        self.assertEqual(particles.dim, 2)
        self.assertEqual(particles.d.shape, (11, 2))
        self.assertTrue(allclose(sum(particles.w), 1.0))

    def test_hypertoroidal_grid_alias_accepts_scalar_resolution(self):
        distribution = HypertoroidalWrappedNormalDistribution(
            array([0.1, 1.0]), 0.05 * eye(2)
        )

        grid = convert_distribution(distribution, "grid", n_grid_points=4)

        self.assertIsInstance(grid, HypertoroidalGridDistribution)
        self.assertEqual(grid.dim, 2)
        self.assertEqual(grid.grid_values.shape, (4, 4))
        self.assertEqual(grid.get_grid().shape, (16, 2))

    def test_hypertoroidal_grid_converts_to_weighted_dirac(self):
        distribution = HypertoroidalWrappedNormalDistribution(
            array([0.1, 1.0]), 0.05 * eye(2)
        )
        grid = distribution.approximate_as("grid", n_grid_points=(3, 4))

        particles = grid.approximate_as("particles")

        self.assertIsInstance(particles, HypertoroidalDiracDistribution)
        self.assertEqual(particles.dim, 2)
        self.assertEqual(particles.d.shape, (12, 2))
        self.assertTrue(allclose(sum(particles.w), 1.0))


if __name__ == "__main__":
    unittest.main()
