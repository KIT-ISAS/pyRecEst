"""Test for uniform distribution for hyperhemispheres"""

import unittest
from unittest.mock import patch

import numpy.testing as npt
import pyrecest.backend

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import (
    all,
    allclose,
    array,
    linalg,
    ones,
    pi,
    random,
    reshape,
    to_numpy,
)
from pyrecest.distributions import HyperhemisphericalUniformDistribution


def get_random_points(n, d):
    random.seed(10)
    points = random.normal(size=(n, d + 1))
    points = points[points[:, -1] >= 0, :]
    points /= reshape(linalg.norm(points, axis=1), (-1, 1))
    return points


class TestHyperhemisphericalUniformDistribution(unittest.TestCase):
    """Test for uniform distribution for hyperhemispheres"""

    def test_pdf_2d(self):
        hhud = HyperhemisphericalUniformDistribution(2)

        points = get_random_points(100, 2)

        self.assertTrue(
            allclose(hhud.pdf(points), ones(points.shape[0]) / (2 * pi), atol=1e-6)
        )

    def test_pdf_3d(self):
        hhud = HyperhemisphericalUniformDistribution(3)

        points = get_random_points(100, 3)
        # jscpd:ignore-start
        self.assertTrue(
            allclose(hhud.pdf(points), ones(points.shape[0]) / (pi**2), atol=1e-6)
        )
        # jscpd:ignore-end

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        "Test not supported for this backend",
    )
    def test_integrate_S2(self):
        hhud = HyperhemisphericalUniformDistribution(2)
        self.assertAlmostEqual(hhud.integrate(), 1.0, delta=1e-6)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        "Test not supported for this backend",
    )
    def test_integrate_S3(self):
        hhud = HyperhemisphericalUniformDistribution(3)
        self.assertAlmostEqual(hhud.integrate(), 1, delta=1e-6)

    @patch(
        "pyrecest.distributions.hypersphere_subset."
        "hyperhemispherical_uniform_distribution.HypersphericalUniformDistribution.sample"
    )
    def test_sample_mirrors_each_point_with_negative_last_coordinate(self, sample_mock):
        raw_samples = array(
            [
                [0.0, 0.0, -1.0],
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        expected_samples = array(
            [
                [-0.0, -0.0, 1.0],
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        sample_mock.return_value = raw_samples

        samples = HyperhemisphericalUniformDistribution(2).sample(4)

        npt.assert_allclose(to_numpy(samples), to_numpy(expected_samples))
        self.assertTrue(bool(to_numpy(all(samples[:, -1] >= 0.0))))

    @patch(
        "pyrecest.distributions.hypersphere_subset."
        "hyperhemispherical_uniform_distribution.HypersphericalUniformDistribution.sample"
    )
    def test_single_sample_is_not_forced_into_positive_orthant(self, sample_mock):
        raw_sample = array([[-0.6, -0.8, 0.0]])
        sample_mock.return_value = raw_sample

        sample = HyperhemisphericalUniformDistribution(2).sample(1)

        npt.assert_allclose(to_numpy(sample), to_numpy(raw_sample))


if __name__ == "__main__":
    unittest.main()
