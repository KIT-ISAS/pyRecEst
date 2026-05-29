import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, diag, eye, isfinite, pi
from pyrecest.distributions import SO3BinghamDistribution
from pyrecest.distributions.hypersphere_subset.hyperhemispherical_bingham_distribution import (
    HyperhemisphericalBinghamDistribution,
)
from tests.distributions.so3_test_helpers import (
    ATOL,
    assert_matches_z_rotation,
    scalar,
    z_quaternion,
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

    def test_constructor_rejects_invalid_parameters(self):
        valid_z = array([-1.0, -1.0, -1.0, 0.0])
        valid_m = eye(4)
        invalid_cases = [
            (array([0.0, 0.0, 0.0]), valid_m, "shape"),
            (valid_z, eye(3), "shape"),
            (array([-1.0, -1.0, float("nan"), 0.0]), valid_m, "finite"),
            (
                valid_z,
                diag(array([1.0, 1.0, 1.0, float("inf")])),
                "finite",
            ),
            (array([-1.0, 0.0, -0.5, 0.0]), valid_m, "ascending"),
            (array([-1.0, -1.0, -1.0, 1.0]), valid_m, "zero"),
            (valid_z, diag(array([1.0, 1.0, 1.0, 2.0])), "orthogonal"),
        ]

        for z, m, message in invalid_cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    SO3BinghamDistribution(z, m)

    def test_factories_reject_invalid_parameters(self):
        for concentration in (-1.0, float("inf"), float("nan")):
            with self.subTest(concentration=concentration):
                with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
                    SO3BinghamDistribution.from_mode_and_concentration(
                        array([0.0, 0.0, 0.0, 1.0]), concentration
                    )

        with self.assertRaisesRegex(ValueError, "shape"):
            SO3BinghamDistribution.calculate_normalization_constant(
                array([0.0, 0.0, 0.0])
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            SO3BinghamDistribution.calculate_normalization_constant(
                array([0.0, 0.0, float("nan"), 0.0])
            )
        with self.assertRaisesRegex(ValueError, "shape"):
            SO3BinghamDistribution.from_concentration_matrix(eye(3))
        with self.assertRaisesRegex(ValueError, "finite"):
            SO3BinghamDistribution.from_concentration_matrix(
                diag(array([0.0, 0.0, float("inf"), 0.0]))
            )
        with self.assertRaisesRegex(ValueError, "BinghamDistribution"):
            SO3BinghamDistribution.from_bingham_distribution(object())

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

        assert_matches_z_rotation(self, dist.mean_rotation_matrix(), pi / 2.0)

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

        with self.assertRaisesRegex(ValueError, "SO3BinghamDistribution"):
            first.multiply(object())
        with self.assertRaisesRegex(ValueError, "SO3BinghamDistribution"):
            first.compose(object())


if __name__ == "__main__":
    unittest.main()
