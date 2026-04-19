import unittest

import numpy as np
import numpy.testing as npt

from pyrecest.utils.point_set_registration import (
    AffineTransform,
    estimate_transform,
    joint_registration_assignment,
    solve_gated_assignment,
)


class TestEstimateTransform(unittest.TestCase):
    def test_estimate_affine_transform_recovers_ground_truth(self):
        source = np.array(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 1.0], [1.5, 2.0]],
            dtype=float,
        )
        true_matrix = np.array([[1.1, 0.2], [-0.15, 0.95]], dtype=float)
        true_offset = np.array([3.0, -1.5], dtype=float)
        target = (true_matrix @ source.T).T + true_offset

        estimated = estimate_transform(source, target, model="affine")

        npt.assert_allclose(estimated.matrix, true_matrix, atol=1e-10)
        npt.assert_allclose(estimated.offset, true_offset, atol=1e-10)

    def test_estimate_rigid_transform_recovers_rotation_and_translation(self):
        source = np.array(
            [[0.0, 0.0], [1.0, 0.2], [0.4, 1.1], [1.3, 1.6], [2.1, -0.3]],
            dtype=float,
        )
        angle = np.deg2rad(25.0)
        true_rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
            dtype=float,
        )
        true_offset = np.array([-0.8, 2.4], dtype=float)
        target = (true_rotation @ source.T).T + true_offset

        estimated = estimate_transform(source, target, model="rigid")

        npt.assert_allclose(estimated.matrix, true_rotation, atol=1e-10)
        npt.assert_allclose(estimated.offset, true_offset, atol=1e-10)


class TestGatedAssignment(unittest.TestCase):
    def test_solve_gated_assignment_leaves_rows_unmatched(self):
        cost_matrix = np.array([[0.1, 10.0], [10.0, 10.0]], dtype=float)
        assignment = solve_gated_assignment(cost_matrix, max_cost=1.0)
        npt.assert_array_equal(assignment, np.array([0, -1], dtype=np.int64))


class TestJointRegistrationAssignment(unittest.TestCase):
    def test_joint_registration_assignment_recovers_permuted_affine_matches(self):
        reference = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1.0],
                [2.0, 0.5],
                [1.5, 1.6],
                [2.3, 1.9],
            ],
            dtype=float,
        )
        true_matrix = np.array([[1.03, 0.08], [-0.04, 0.97]], dtype=float)
        true_offset = np.array([4.0, -2.0], dtype=float)
        transformed = (true_matrix @ reference.T).T + true_offset
        permutation = np.array([3, 5, 1, 4, 0, 2])
        moving = transformed[permutation]

        result = joint_registration_assignment(
            reference,
            moving,
            model="affine",
            max_cost=5.0,
            tolerance=1e-12,
        )

        expected_assignment = np.empty(reference.shape[0], dtype=np.int64)
        expected_assignment[permutation] = np.arange(reference.shape[0], dtype=np.int64)
        npt.assert_array_equal(result.assignment, expected_assignment)
        npt.assert_allclose(result.transform.matrix, true_matrix, atol=1e-10)
        npt.assert_allclose(result.transform.offset, true_offset, atol=1e-10)
        self.assertTrue(result.converged)

    def test_joint_registration_assignment_handles_missing_points_and_outliers(self):
        reference = np.array(
            [[0.0, 0.0], [1.0, 0.0], [0.3, 0.9], [1.1, 1.2], [2.0, -0.2]],
            dtype=float,
        )
        shift = np.array([6.0, -3.0], dtype=float)
        moving = reference[[1, 3, 4]] + shift
        moving = np.vstack([moving, np.array([[50.0, 50.0]], dtype=float)])

        result = joint_registration_assignment(
            reference,
            moving,
            model="translation",
            initial_transform=AffineTransform(
                np.eye(2, dtype=float), shift + np.array([0.1, -0.1], dtype=float)
            ),
            max_cost=0.35,
        )

        self.assertEqual(np.sum(result.assignment >= 0), 3)
        self.assertTrue(np.all(result.assignment[[0, 2]] == -1))
        npt.assert_allclose(result.transform.offset, shift, atol=1e-10)

    def test_joint_registration_assignment_supports_custom_cost_function(self):
        reference = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=float)
        moving = reference + np.array([0.5, -0.2], dtype=float)
        preference = np.array(
            [
                [0.0, 10.0, 10.0],
                [10.0, 0.0, 10.0],
                [10.0, 10.0, 0.0],
            ],
            dtype=float,
        )

        def cost_function(transformed_reference, moving_points):
            del transformed_reference, moving_points
            return preference

        result = joint_registration_assignment(
            reference,
            moving,
            model="translation",
            cost_function=cost_function,
            max_cost=1.0,
        )

        npt.assert_array_equal(result.assignment, np.array([0, 1, 2], dtype=np.int64))


if __name__ == "__main__":
    unittest.main()
