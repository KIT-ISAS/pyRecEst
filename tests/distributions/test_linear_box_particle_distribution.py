import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, ones
from pyrecest.distributions.nonperiodic.linear_box_particle_distribution import (
    LinearBoxParticleDistribution,
)


class LinearBoxParticleDistributionTest(unittest.TestCase):
    def test_mean_and_covariance_include_uniform_box_variance(self):
        dist = LinearBoxParticleDistribution(array([[0.0, 0.0]]), array([[2.0, 4.0]]))

        npt.assert_allclose(dist.mean(), array([1.0, 2.0]))
        npt.assert_allclose(
            dist.covariance(), array([[1.0 / 3.0, 0.0], [0.0, 4.0 / 3.0]])
        )

    def test_pdf_of_single_one_dimensional_box(self):
        dist = LinearBoxParticleDistribution(array([[0.0]]), array([[2.0]]))

        npt.assert_allclose(
            dist.pdf(array([-1.0, 0.5, 1.5, 3.0])), array([0.0, 0.5, 0.5, 0.0])
        )

    def test_sample_rejects_invalid_count(self):
        dist = LinearBoxParticleDistribution(array([[0.0]]), array([[2.0]]))

        for n in (0, -1, 1.5, True):
            with self.subTest(n=n):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    dist.sample(n)

    def test_integrate_query_box(self):
        dist = LinearBoxParticleDistribution(array([[0.0]]), array([[4.0]]), ones(1))

        npt.assert_allclose(dist.integrate(array([1.0]), array([3.0])), 0.5)

    def test_from_distribution_rejects_invalid_particle_count_aliases(self):
        for n_particles in (0, -1, 1.5, True):
            with self.subTest(n_particles=n_particles):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    LinearBoxParticleDistribution.from_distribution(
                        object(), n_particles=n_particles
                    )


if __name__ == "__main__":
    unittest.main()
