import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, cos, diag, eye, isfinite, pi, sin, to_numpy
from pyrecest.distributions import SO3BinghamDistribution
from pyrecest.distributions.hypersphere_subset.hyperhemispherical_bingham_distribution import (
    HyperhemisphericalBinghamDistribution,
)

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


class SO3BinghamDistributionTest(unittest.TestCase):
    def test_inherits_hyperhemispherical_bingham_distribution(self):
        dist = SO3BinghamDistribution.from_mode_and_concentration(
            array([0.0, 0.0, 0.0, 1.0]), 3.0
        )

        self.assertIsInstance(dist, HyperhemisphericalBinghamDistribution)
        self.assertTrue(dist.is_valid())

    def test_from_mode_and_concentration_canonicalizes_mode(self):
        dist = SO3BinghamDistribution.from_mode_and_concentration(
            array([0.0, 0.0, 0.0, -2.0]), 5.0
        )

        npt.assert_allclose(dist.mode(), array([0.0, 0.0, 0.0, 1.0]), atol=ATOL)
        npt.assert_allclose(
            dist.distFullSphere.Z, array([-5.0, -5.0, -5.0, 0.0]), atol=ATOL
        )

    def test_pdf_is_antipodally_symmetric_and_peaks_at_mode(self):
        dist = SO3BinghamDistribution.from_mode_and_concentration(
            array([0.0, 0.0, 0.0, 1.0]), 4.0
        )
        mode = array([0.0, 0.0, 0.0, 1.0])
        antipodal_mode = -mode
        side = array([1.0, 0.0, 0.0, 0.0])

        mode_pdf = scalar(dist.pdf(mode))
        side_pdf = scalar(dist.pdf(side))

        self.assertGreater(mode_pdf, side_pdf)
        npt.assert_allclose(mode_pdf, scalar(dist.pdf(antipodal_mode)), atol=ATOL)
        self.assertTrue(bool(isfinite(dist.pdf(mode))))

    def test_from_concentration_matrix_roundtrip(self):
        concentration_matrix = diag(array([-2.0, -2.0, -2.0, 0.0]))

        dist = SO3BinghamDistribution.from_concentration_matrix(concentration_matrix)

        npt.assert_allclose(dist.mode(), array([0.0, 0.0, 0.0, 1.0]), atol=ATOL)
        npt.assert_allclose(
            dist.concentration_matrix(), concentration_matrix, atol=ATOL
        )

    def test_geodesic_distance_respects_antipodal_equivalence(self):
        identity = array([0.0, 0.0, 0.0, 1.0])
        identity_antipodal = array([0.0, 0.0, 0.0, -1.0])
        quarter_turn = z_quaternion(pi / 2.0)

        npt.assert_allclose(
            SO3BinghamDistribution.geodesic_distance(identity, identity_antipodal),
            array([0.0]),
            atol=ATOL,
        )
        npt.assert_allclose(
            SO3BinghamDistribution.geodesic_distance(identity, quarter_turn),
            array([pi / 2.0]),
            atol=ATOL,
        )

    def test_mean_rotation_matrix_matches_mode_quaternion(self):
        dist = SO3BinghamDistribution.from_mode_and_concentration(
            z_quaternion(pi / 2.0), 3.0
        )

        npt.assert_allclose(
            dist.mean_rotation_matrix(), z_rotation(pi / 2.0), atol=ATOL
        )
        npt.assert_allclose(
            dist.mean_rotation_matrix().T @ dist.mean_rotation_matrix(),
            eye(3),
            atol=ATOL,
        )

    def test_multiply_returns_so3_bingham_distribution(self):
        first = SO3BinghamDistribution.from_mode_and_concentration(
            array([0.0, 0.0, 0.0, 1.0]), 2.0
        )
        second = SO3BinghamDistribution.from_mode_and_concentration(
            array([0.0, 0.0, 0.0, 1.0]), 3.0
        )

        product = first.multiply(second)

        self.assertIsInstance(product, SO3BinghamDistribution)
        npt.assert_allclose(product.mode(), array([0.0, 0.0, 0.0, 1.0]), atol=ATOL)
        npt.assert_allclose(
            product.distFullSphere.Z, array([-5.0, -5.0, -5.0, 0.0]), atol=ATOL
        )


if __name__ == "__main__":
    unittest.main()
