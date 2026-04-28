import math
import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import (
    array,
    cos,
    diag,
    eye,
    linalg,
    ones,
    pi,
    random,
    sin,
    sqrt,
    to_numpy,
)
from pyrecest.distributions import SO3TangentGaussianDistribution

ATOL = 1e-6


def scalar(value):
    return float(to_numpy(value).reshape(-1)[0])


def z_quaternion(angle):
    return array([0.0, 0.0, sin(angle / 2.0), cos(angle / 2.0)])


def z_rotation(angle):
    return array(
        [
            [cos(angle), -sin(angle), 0.0],
            [sin(angle), cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


class SO3TangentGaussianDistributionTest(unittest.TestCase):
    def test_constructor_normalizes_and_canonicalizes_mean(self):
        covariance = diag(array([0.1, 0.2, 0.3]))
        dist = SO3TangentGaussianDistribution(array([0.0, 0.0, 0.0, -2.0]), covariance)

        npt.assert_allclose(dist.mean(), array([0.0, 0.0, 0.0, 1.0]), atol=ATOL)
        npt.assert_allclose(dist.covariance(), covariance, atol=ATOL)
        self.assertTrue(dist.is_valid())

    def test_exp_log_roundtrip_with_base_rotation(self):
        base = z_quaternion(pi / 3.0)
        tangent_vectors = array([[0.1, -0.2, 0.05], [0.0, 0.0, 0.0]])

        rotations = SO3TangentGaussianDistribution.exp_map(tangent_vectors, base=base)
        roundtrip = SO3TangentGaussianDistribution.log_map(rotations, base=base)

        npt.assert_allclose(roundtrip, tangent_vectors, atol=ATOL)

    def test_pdf_and_ln_pdf_peak_at_mode(self):
        covariance = diag(array([0.2, 0.3, 0.4]))
        dist = SO3TangentGaussianDistribution(array([0.0, 0.0, 0.0, 1.0]), covariance)
        offset = SO3TangentGaussianDistribution.exp_map(array([0.4, 0.0, 0.0]))

        mode_pdf = scalar(dist.pdf(dist.mode()))
        offset_pdf = scalar(dist.pdf(offset))
        expected_mode_pdf = 1.0 / scalar(sqrt((2.0 * pi) ** 3 * linalg.det(covariance)))

        self.assertGreater(mode_pdf, offset_pdf)
        npt.assert_allclose(mode_pdf, expected_mode_pdf, atol=ATOL)
        npt.assert_allclose(
            scalar(dist.ln_pdf(dist.mode())), math.log(mode_pdf), atol=ATOL
        )

    def test_sampling_returns_unit_quaternions(self):
        random.seed(0)
        dist = SO3TangentGaussianDistribution.from_covariance_diagonal(
            array([0.0, 0.0, 0.0, 1.0]), array([0.01, 0.01, 0.01])
        )

        samples = dist.sample(8)

        self.assertEqual(samples.shape, (8, 4))
        npt.assert_allclose(linalg.norm(samples, None, -1), ones(8), atol=ATOL)

    def test_geodesic_distance_respects_antipodal_equivalence(self):
        identity = array([0.0, 0.0, 0.0, 1.0])
        identity_antipodal = array([0.0, 0.0, 0.0, -1.0])
        quarter_turn = z_quaternion(pi / 2.0)

        npt.assert_allclose(
            SO3TangentGaussianDistribution.geodesic_distance(
                identity, identity_antipodal
            ),
            array([0.0]),
            atol=ATOL,
        )
        npt.assert_allclose(
            SO3TangentGaussianDistribution.geodesic_distance(identity, quarter_turn),
            array([pi / 2.0]),
            atol=ATOL,
        )

    def test_mean_rotation_matrix_matches_quaternion(self):
        dist = SO3TangentGaussianDistribution(
            z_quaternion(pi / 2.0), diag(array([0.1, 0.1, 0.1]))
        )

        npt.assert_allclose(
            dist.mean_rotation_matrix(), z_rotation(pi / 2.0), atol=ATOL
        )
        npt.assert_allclose(
            dist.mean_rotation_matrix().T @ dist.mean_rotation_matrix(),
            eye(3),
            atol=ATOL,
        )


if __name__ == "__main__":
    unittest.main()
