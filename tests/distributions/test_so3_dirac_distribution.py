import unittest

import numpy as np
import numpy.testing as npt
import pyrecest.backend

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, cos, eye, linalg, pi, sin, sqrt, stack
from pyrecest.distributions import SO3DiracDistribution, SO3UniformDistribution
from pyrecest.distributions.hypersphere_subset.hyperhemispherical_dirac_distribution import (
    HyperhemisphericalDiracDistribution,
)

ATOL = 1e-6


class SO3DiracDistributionTest(unittest.TestCase):
    def test_inherits_hyperhemispherical_dirac_distribution(self):
        dist = SO3DiracDistribution(array([0.0, 0.0, 0.0, 1.0]))

        self.assertIsInstance(dist, HyperhemisphericalDiracDistribution)

    def test_constructor_normalizes_and_canonicalizes_quaternions(self):
        dist = SO3DiracDistribution(
            array([[0.0, 0.0, 0.0, -2.0], [0.0, 0.0, 0.0, 3.0]]),
            array([2.0 / 3.0, 1.0 / 3.0]),
        )

        npt.assert_allclose(dist.d[0], array([0.0, 0.0, 0.0, 1.0]), atol=ATOL)
        npt.assert_allclose(dist.d[1], array([0.0, 0.0, 0.0, 1.0]), atol=ATOL)
        npt.assert_allclose(dist.w, array([2.0 / 3.0, 1.0 / 3.0]), atol=ATOL)
        self.assertTrue(dist.is_valid())

    def test_antipodal_quaternions_have_same_mean(self):
        angle = pi / 3.0
        quat = array([0.0, 0.0, sin(angle / 2.0), cos(angle / 2.0)])
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

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "pytorch",
        reason="Rotation matrix conversion is not supported on the PyTorch backend.",
    )
    def test_from_rotation_matrices_rejects_invalid_inputs(self):
        invalid_cases = [
            (array([1.0, 0.0, 0.0]), "shape"),
            (array([[1.0, 0.0], [0.0, 1.0]]), "shape"),
            (
                array(
                    [
                        [1.0, 0.0, 0.0],
                        [0.0, float("nan"), 0.0],
                        [0.0, 0.0, 1.0],
                    ]
                ),
                "finite",
            ),
        ]

        for rotation_matrices, message in invalid_cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    SO3DiracDistribution.from_rotation_matrices(rotation_matrices)

    def test_from_distribution_validates_particle_count(self):
        source = SO3UniformDistribution()

        dist = SO3DiracDistribution.from_distribution(source, np.int64(3))

        self.assertEqual(dist.d.shape, (3, 4))
        self.assertEqual(dist.w.shape, (3,))
        self.assertTrue(dist.is_valid())

        for n_particles in (True, 1.5, 0, -1, None):
            with self.subTest(n_particles=n_particles):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    SO3DiracDistribution.from_distribution(source, n_particles)

    def test_geodesic_distance_respects_antipodal_equivalence(self):
        identity = array([0.0, 0.0, 0.0, 1.0])
        identity_antipodal = array([0.0, 0.0, 0.0, -1.0])
        quarter_turn = array([0.0, 0.0, sin(pi / 4.0), cos(pi / 4.0)])

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

    def test_chordal_distance_matches_so3_frobenius_metric(self):
        identity = array([0.0, 0.0, 0.0, 1.0])
        identity_antipodal = array([0.0, 0.0, 0.0, -1.0])
        quarter_turn = array([0.0, 0.0, sin(pi / 4.0), cos(pi / 4.0)])
        half_turn = array([0.0, 0.0, 1.0, 0.0])
        quarter_turn_matrix = array(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )

        npt.assert_allclose(
            SO3DiracDistribution.chordal_distance(identity, identity_antipodal),
            array([0.0]),
            atol=ATOL,
        )
        npt.assert_allclose(
            SO3DiracDistribution.chordal_distance(identity, quarter_turn),
            linalg.norm(eye(3) - quarter_turn_matrix),
            atol=ATOL,
        )
        npt.assert_allclose(
            SO3DiracDistribution.chordal_distance(identity, half_turn),
            array([2.0 * sqrt(2.0)]),
            atol=ATOL,
        )

    def test_so3_distance_methods_accept_batches(self):
        identity = array([0.0, 0.0, 0.0, 1.0])
        quarter_turn = array([0.0, 0.0, sin(pi / 4.0), cos(pi / 4.0)])
        dist = SO3DiracDistribution(
            stack([identity, quarter_turn], axis=0), array([0.25, 0.75])
        )

        npt.assert_allclose(
            dist.distance_to(identity), array([0.0, pi / 2.0]), atol=ATOL
        )
        npt.assert_allclose(
            dist.chordal_distance_to(identity), array([0.0, 2.0]), atol=ATOL
        )
        npt.assert_allclose(dist.angular_error_mean(identity), 0.75 * pi / 2.0)
        npt.assert_allclose(dist.chordal_error_mean(identity), 1.5, atol=ATOL)

    def test_mode_returns_highest_weight_canonical_quaternion(self):
        dist = SO3DiracDistribution(
            array([[0.0, 0.0, 0.0, 1.0], [0.5, 0.5, 0.5, -0.5]]),
            array([0.1, 0.9]),
        )

        npt.assert_allclose(dist.mode(), array([-0.5, -0.5, -0.5, 0.5]), atol=ATOL)


if __name__ == "__main__":
    unittest.main()
