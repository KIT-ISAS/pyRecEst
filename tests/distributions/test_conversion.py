import unittest
import warnings

# pylint: disable=no-name-in-module,no-member
import pyrecest
from pyrecest.backend import all as backend_all
from pyrecest.backend import allclose, array, eye, linalg, random, sum
from pyrecest.distributions import GaussianDistribution, LinearDiracDistribution
from pyrecest.distributions.conversion import (
    ConversionError,
    ConversionResult,
    can_convert,
    convert_distribution,
    register_conversion_alias,
)
from pyrecest.distributions.hypersphere_subset.hyperhemispherical_dirac_distribution import (
    HyperhemisphericalDiracDistribution,
)
from pyrecest.distributions.hypersphere_subset.hyperhemispherical_grid_distribution import (
    HyperhemisphericalGridDistribution,
)
from pyrecest.distributions.hypersphere_subset.hyperhemispherical_uniform_distribution import (
    HyperhemisphericalUniformDistribution,
)
from pyrecest.distributions.hypersphere_subset.hyperspherical_dirac_distribution import (
    HypersphericalDiracDistribution,
)
from pyrecest.distributions.hypersphere_subset.hyperspherical_grid_distribution import (
    HypersphericalGridDistribution,
)
from pyrecest.distributions.hypersphere_subset.von_mises_fisher_distribution import (
    VonMisesFisherDistribution,
)
from pyrecest.distributions.hypersphere_subset.watson_distribution import (
    WatsonDistribution,
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

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ != "numpy",  # pylint: disable=no-member
        reason="vMF sampling-based conversion currently requires the NumPy backend.",
    )
    def test_hyperspherical_particles_alias(self):
        random.seed(0)
        vmf = VonMisesFisherDistribution(array([1.0, 0.0, 0.0]), 5.0)

        particles = convert_distribution(vmf, "particles", n_particles=20)

        self.assertIsInstance(particles, HypersphericalDiracDistribution)
        self.assertEqual(particles.d.shape, (20, 3))
        self.assertTrue(allclose(sum(particles.w), 1.0))
        self.assertTrue(allclose(linalg.norm(particles.d, axis=1), 1.0, atol=1e-10))

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Leopardi grid generation is not supported on this backend.",
    )
    def test_hyperspherical_grid_alias_uses_default_grid_type(self):
        mu = array([1.0, 0.0, 0.0])
        vmf = VonMisesFisherDistribution(mu, 5.0)

        grid = convert_distribution(vmf, "grid", no_of_grid_points=84)

        self.assertIsInstance(grid, HypersphericalGridDistribution)
        self.assertEqual(grid.grid_type, "leopardi")
        self.assertEqual(grid.get_grid().shape[1], vmf.input_dim)
        self.assertTrue(allclose(linalg.norm(grid.get_grid(), axis=1), 1.0, atol=1e-10))
        self.assertTrue(allclose(grid.mean_direction(), mu, atol=0.35))

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Leopardi grid generation is not supported on this backend.",
    )
    def test_hyperspherical_grid_to_particles_alias_uses_weighted_grid(self):
        vmf = VonMisesFisherDistribution(array([1.0, 0.0, 0.0]), 5.0)
        grid = convert_distribution(vmf, "grid", no_of_grid_points=84)

        particles = convert_distribution(grid, "particles")

        self.assertIsInstance(particles, HypersphericalDiracDistribution)
        self.assertEqual(particles.d.shape, grid.get_grid().shape)
        self.assertTrue(allclose(particles.d, grid.get_grid()))
        self.assertTrue(allclose(sum(particles.w), 1.0))
        self.assertTrue(allclose(particles.mean_direction(), grid.mean_direction()))

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Leopardi grid generation is not supported on this backend.",
    )
    def test_hyperhemispherical_grid_alias_uses_default_grid_type(self):
        uniform = HyperhemisphericalUniformDistribution(2)

        grid = convert_distribution(uniform, "grid", no_of_grid_points=40)

        self.assertIsInstance(grid, HyperhemisphericalGridDistribution)
        self.assertEqual(grid.grid_type, "leopardi_symm")
        self.assertEqual(grid.get_grid().shape[1], uniform.input_dim)
        self.assertTrue(backend_all(grid.get_grid()[:, -1] >= -1e-12))

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Leopardi grid generation is not supported on this backend.",
    )
    def test_watson_to_hyperhemispherical_grid_alias(self):
        watson = WatsonDistribution(array([1.0, 0.0, 0.0]), 1.0)

        grid = convert_distribution(
            watson, "hyperhemispherical_grid", no_of_grid_points=40
        )

        self.assertIsInstance(grid, HyperhemisphericalGridDistribution)
        self.assertEqual(grid.grid_type, "leopardi_symm")
        self.assertEqual(grid.get_grid().shape[1], watson.input_dim)
        self.assertTrue(backend_all(grid.get_grid()[:, -1] >= -1e-12))

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Leopardi grid generation is not supported on this backend.",
    )
    def test_hyperhemispherical_grid_to_particles_alias_uses_weighted_grid(self):
        uniform = HyperhemisphericalUniformDistribution(2)
        grid = convert_distribution(uniform, "grid", no_of_grid_points=40)

        particles = convert_distribution(grid, "particles")

        self.assertIsInstance(particles, HyperhemisphericalDiracDistribution)
        self.assertEqual(particles.d.shape, grid.get_grid().shape)
        self.assertTrue(allclose(particles.d, grid.get_grid()))
        self.assertTrue(allclose(sum(particles.w), 1.0))
        self.assertTrue(backend_all(particles.d[:, -1] >= -1e-12))

    def test_hyperhemispherical_grid_mean_direction_uses_row_indexing(self):
        grid = HyperhemisphericalGridDistribution(
            array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]), array([0.1, 0.9])
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mode = grid.mean_direction()

        self.assertTrue(allclose(mode, array([0.0, 0.0, 1.0])))


if __name__ == "__main__":
    unittest.main()
