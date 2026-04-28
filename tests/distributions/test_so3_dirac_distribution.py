import unittest

import numpy.testing as npt
import pyrecest.backend

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, cos, eye, pi, sin, stack
from pyrecest.distributions import SO3DiracDistribution

ATOL = 1e-6


class SO3DiracDistributionTest(unittest.TestCase):
    def test_constructor_normalizes_and_canonicalizes_quaternions(self):
        dist = SO3DiracDistribution(
            array([[-2.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 3.0]]),
            array([2.0 / 3.0, 1.0 / 3.0]),
        )

        npt.assert_allclose(dist.d[0], array([1.0, 0.0, 0.0, 0.0]), atol=ATOL)
        npt.assert_allclose(dist.d[1], array([0.0, 0.0, 0.0, 1.0]), atol=ATOL)
        npt.assert_allclose(dist.w, array([2.0 / 3.0, 1.0 / 3.0]), atol=ATOL)
        self.assertTrue(dist.is_valid())

    def test_antipodal_quaternions_have_same_mean(self):
        angle = pi / 3.0
        quat = array([cos(angle / 2.0), 0.0, 0.0, sin(angle / 2.0)])
        dist = SO3DiracDistribution(stack([quat, -quat], axis=0))

        npt.assert_allclose(dist.mean(), quat, atol=ATOL)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "pytorch",
        reason="Rotation matrix conversion is not supported on the PyTorch backend.",
    )
    def test_rotation_matrix_roundtrip(self):
        quarter_turn_z = array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        rotations = stack([eye(3), quarter_turn_z], axis=0)

        dist = SO3DiracDistribution.from_rotation_matrices(rotations)

        npt.assert_allclose(dist.as_rotation_matrices(), rotations, atol=ATOL)

    def test_geodesic_distance_respects_antipodal_equivalence(self):
        identity = array([1.0, 0.0, 0.0, 0.0])
        identity_antipodal = array([-1.0, 0.0, 0.0, 0.0])
        quarter_turn = array([cos(pi / 4.0), 0.0, 0.0, sin(pi / 4.0)])

        npt.assert_allclose(
            SO3DiracDistribution.geodesic_distance(identity, identity_antipodal),
            array([0.0]),
            atol=ATOL,
        )
        npt.assert_allclose(
            SO3DiracDistribution.geodesic_distance(identity, quarter_turn),
            array([pi / 2.0]),
            atol=ATOL,
        )

    def test_mode_returns_highest_weight_canonical_quaternion(self):
        dist = SO3DiracDistribution(
            array([[1.0, 0.0, 0.0, 0.0], [-0.5, 0.5, 0.5, 0.5]]),
            array([0.1, 0.9]),
        )

        npt.assert_allclose(dist.mode(), array([0.5, -0.5, -0.5, -0.5]), atol=ATOL)


if __name__ == "__main__":
    unittest.main()
