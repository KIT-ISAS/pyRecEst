import unittest

import numpy as np
import scipy
from parameterized import parameterized
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters import KalmanFilter
from pyrecest.filters.global_nearest_neighbor import GlobalNearestNeighbor


class GlobalNearestNeighborTest(unittest.TestCase):
    """Test case for the GlobalNearestNeighbor class."""

    def setUp(self):
        """Initialize test variables before each test is run."""
        self.kfs_init = [
            KalmanFilter(GaussianDistribution(np.zeros(4), np.diag([1, 2, 3, 4]))),
            KalmanFilter(
                GaussianDistribution(np.array([1, 2, 3, 4]), np.diag([2, 2, 2, 2]))
            ),
            KalmanFilter(
                GaussianDistribution(-np.array([1, 2, 3, 4]), np.diag([4, 3, 2, 1]))
            ),
        ]
        self.meas_mat = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
        self.sys_mat = scipy.linalg.block_diag([[1, 1], [0, 1]], [[1, 1], [0, 1]])
        self.all_different_meas_covs = np.dstack(
            [
                np.diag([1, 2]),
                np.array([[5, 0.1], [0.1, 3]]),
                np.array([[2, -0.5], [-0.5, 0.5]]),
            ]
        )
        self.all_different_meas_covs_4 = np.dstack(
            (self.all_different_meas_covs, np.array([[2, -0.5], [-0.5, 0.5]]))
        )

    def test_set_state_sets_correct_state(self):
        tracker = GlobalNearestNeighbor()
        tracker.filter_state = self.kfs_init
        self.assertEqual(
            len(tracker.filter_state),
            len(self.kfs_init),
            "State was not set correctly.",
        )

    def test_get_state_returns_correct_shape(self):
        tracker = GlobalNearestNeighbor()
        tracker.filter_state = self.kfs_init
        self.assertEqual(tracker.get_point_estimate().shape, (4, 3))
        self.assertEqual(tracker.get_point_estimate(True).shape, (12,))

    @parameterized.expand(
        [("no_inputs", np.zeros(4)), ("with_inputs", np.array([1, -1, 1, -1]))]
    )
    def test_predict_linear(self, name, sys_input):
        C_matrices = [
            scipy.linalg.block_diag([[3, 2], [2, 2]], [[7, 4], [4, 4]]) + np.eye(4),
            scipy.linalg.block_diag([[4, 2], [2, 2]], [[4, 2], [2, 2]]) + np.eye(4),
            scipy.linalg.block_diag([[7, 3], [3, 3]], [[3, 1], [1, 1]]) + np.eye(4),
        ]

        tracker = GlobalNearestNeighbor()
        tracker.filter_state = self.kfs_init
        if name == "no_inputs":
            tracker.predict_linear(self.sys_mat, np.eye(4))
        else:
            tracker.predict_linear(self.sys_mat, np.eye(4), sys_input)

        for i in range(3):
            with self.subTest(i=i):
                np.testing.assert_array_equal(
                    tracker.filter_bank[i].get_point_estimate(),
                    self.sys_mat @ self.kfs_init[i].get_point_estimate() + sys_input,
                )
                np.testing.assert_array_equal(
                    tracker.filter_bank[i].filter_state.C, C_matrices[i]
                )

    def test_predict_linear_different_mats_and_inputs(self):
        tracker = GlobalNearestNeighbor()
        tracker.filter_state = self.kfs_init

        sys_mats = np.dstack(
            (
                scipy.linalg.block_diag([[1, 1], [0, 1]], [[1, 1], [0, 1]]),
                np.eye(4),
                np.array([[0, 0, 1, 1], [0, 0, 0, 1], [1, 1, 0, 0], [0, 1, 0, 0]]),
            )
        )
        sys_noises = np.dstack(
            (np.eye(4), np.diag([10, 11, 12, 13]), np.diag([1, 5, 3, 5]))
        )
        sys_inputs = np.array([[-1, 1, -1, 1], [1, 2, 3, 4], -np.array([4, 3, 2, 1])]).T

        tracker.predict_linear(sys_mats, sys_noises, sys_inputs)

        np.testing.assert_array_equal(
            tracker.filter_bank[0].filter_state.mu, np.array([-1, 1, -1, 1])
        )
        np.testing.assert_array_equal(
            tracker.filter_bank[1].filter_state.mu, np.array([2, 4, 6, 8])
        )
        np.testing.assert_array_equal(
            tracker.filter_bank[2].filter_state.mu, np.array([-11, -7, -5, -3])
        )
        np.testing.assert_array_equal(
            tracker.filter_bank[0].filter_state.C,
            scipy.linalg.block_diag([[4, 2], [2, 3]], [[8, 4], [4, 5]]),
        )
        np.testing.assert_array_equal(
            tracker.filter_bank[1].filter_state.C, np.diag([12, 13, 14, 15])
        )
        np.testing.assert_array_equal(
            tracker.filter_bank[2].filter_state.C,
            scipy.linalg.block_diag([[4, 1], [1, 6]], [[10, 3], [3, 8]]),
        )

    def test_association_no_clutter(self):
        tracker = GlobalNearestNeighbor()
        tracker.filter_state = self.kfs_init
        all_gaussians = [kf.filter_state for kf in self.kfs_init]

        # Generate perfect measurements, association should then be
        # optimal.
        perfect_meas_ordered = (
            self.meas_mat @ np.array([gaussian.mu for gaussian in all_gaussians]).T
        )
        association = tracker.find_association(
            perfect_meas_ordered, self.meas_mat, np.eye(2)
        )
        np.testing.assert_array_equal(association, [0, 1, 2])

        # Shift them
        measurements = np.roll(perfect_meas_ordered, 1, axis=1)
        association = tracker.find_association(measurements, self.meas_mat, np.eye(2))
        np.testing.assert_array_equal(
            measurements[:, association], perfect_meas_ordered
        )

        # Shift them and add a bit of noise
        measurements = np.roll(perfect_meas_ordered, 1, axis=1) + 0.1
        association = tracker.find_association(measurements, self.meas_mat, np.eye(2))
        np.testing.assert_array_equal(
            measurements[:, association], perfect_meas_ordered + 0.1
        )

        # Use different covariances
        association = tracker.find_association(
            np.roll(perfect_meas_ordered, 1, axis=1) + 0.1,
            self.meas_mat,
            self.all_different_meas_covs,
        )
        np.testing.assert_array_equal(
            measurements[:, association], perfect_meas_ordered + 0.1
        )

    def test_association_with_clutter(self):
        tracker = GlobalNearestNeighbor()
        tracker.filter_state = self.kfs_init
        all_gaussians = [kf.filter_state for kf in self.kfs_init]

        # Generate perfect measurements, association should then be
        # optimal.
        perfect_meas_ordered = self.meas_mat @ np.column_stack(
            [gaussian.mu for gaussian in all_gaussians]
        )
        measurements = np.column_stack([perfect_meas_ordered, np.array([3, 2])])
        association = tracker.find_association(measurements, self.meas_mat, np.eye(2))
        np.testing.assert_array_equal(association, [0, 1, 2])

        # Shift them and add one measurement
        measurements = np.column_stack(
            [
                perfect_meas_ordered[:, 1],
                perfect_meas_ordered[:, 2],
                np.array([2, 2]),
                perfect_meas_ordered[:, 0],
            ]
        )
        association = tracker.find_association(measurements, self.meas_mat, np.eye(2))
        np.testing.assert_array_equal(
            measurements[:, association], perfect_meas_ordered
        )

        # Shift them, add one add one meausurement, and add a bit of noise
        association = tracker.find_association(
            measurements + 0.1, self.meas_mat, np.eye(2)
        )
        np.testing.assert_array_equal(
            measurements[:, association] + 0.1, perfect_meas_ordered + 0.1
        )

        # Use different covariances
        association = tracker.find_association(
            measurements + 0.1, self.meas_mat, self.all_different_meas_covs_4
        )
        np.testing.assert_array_equal(
            measurements[:, association] + 0.1, perfect_meas_ordered + 0.1
        )

    def test_update_with_and_without_clutter(self):
        tracker_no_clut = GlobalNearestNeighbor()
        tracker_clut = GlobalNearestNeighbor()
        tracker_no_clut.filter_state = self.kfs_init
        tracker_clut.filter_state = self.kfs_init
        all_gaussians = [kf.filter_state for kf in self.kfs_init]

        perfect_meas_ordered = self.meas_mat @ np.column_stack(
            [gaussian.mu for gaussian in all_gaussians]
        )
        measurements_no_clut = perfect_meas_ordered
        tracker_no_clut.update_linear(measurements_no_clut, self.meas_mat, np.eye(2))

        self.assertTrue(
            np.allclose(
                [dist.mu for dist in tracker_no_clut.filter_state],
                [dist.mu for dist in all_gaussians],
            )
        )
        curr_covs = np.dstack([dist.C for dist in tracker_no_clut.filter_state])
        self.assertTrue(
            np.all(curr_covs <= np.dstack([dist.C for dist in all_gaussians]))
        )

        measurements_clut = np.column_stack(
            [measurements_no_clut, np.array([2, 2]).reshape(-1, 1)]
        )
        tracker_clut.update_linear(measurements_clut, self.meas_mat, np.eye(2))
        self.assertTrue(
            np.allclose(
                tracker_clut.get_point_estimate(), tracker_no_clut.get_point_estimate()
            )
        )

        measurements_no_clut = perfect_meas_ordered[:, [1, 2, 0]]
        tracker_no_clut.update_linear(measurements_no_clut, self.meas_mat, np.eye(2))
        self.assertTrue(
            np.allclose(
                [dist.mu for dist in tracker_no_clut.filter_state],
                [dist.mu for dist in all_gaussians],
            )
        )
        previous_covs = curr_covs
        curr_covs = np.dstack([dist.C for dist in tracker_no_clut.filter_state])
        self.assertTrue(np.all(curr_covs <= previous_covs))

        measurements_clut = np.column_stack(
            [
                perfect_meas_ordered[:, [1, 2]],
                np.array([2, 2]).reshape(-1, 1),
                perfect_meas_ordered[:, 0],
            ]
        )
        tracker_clut.update_linear(measurements_clut, self.meas_mat, np.eye(2))
        self.assertTrue(
            np.allclose(
                tracker_clut.get_point_estimate(), tracker_no_clut.get_point_estimate()
            )
        )

        measurements_no_clut += 0.1
        tracker_no_clut.update_linear(measurements_no_clut, self.meas_mat, np.eye(2))
        curr_means = [dist.mu for dist in tracker_no_clut.filter_state]
        self.assertFalse(np.allclose(curr_means, [dist.mu for dist in all_gaussians]))
        previous_covs = curr_covs
        curr_covs = np.dstack([dist.C for dist in tracker_no_clut.filter_state])
        self.assertTrue(np.all(curr_covs <= previous_covs))

        measurements_clut += 0.1
        tracker_clut.update_linear(measurements_clut, self.meas_mat, np.eye(2))
        self.assertTrue(
            np.allclose(
                tracker_clut.get_point_estimate(), tracker_no_clut.get_point_estimate()
            )
        )

        tracker_no_clut.update_linear(
            measurements_no_clut, self.meas_mat, self.all_different_meas_covs
        )
        previous_means = curr_means
        self.assertFalse(
            np.allclose(
                [dist.mu for dist in tracker_no_clut.filter_state], previous_means
            )
        )
        previous_covs = curr_covs
        curr_covs = np.dstack([dist.C for dist in tracker_no_clut.filter_state])
        for i in range(curr_covs.shape[2]):
            self.assertTrue(
                np.all(
                    np.sort(np.real(scipy.linalg.eigvals(curr_covs[:, :, i])))
                    <= np.sort(np.real(scipy.linalg.eigvals(previous_covs[:, :, i])))
                )
            )

        tracker_clut.update_linear(
            measurements_clut, self.meas_mat, self.all_different_meas_covs_4
        )
        self.assertTrue(
            np.allclose(
                tracker_clut.get_point_estimate(), tracker_no_clut.get_point_estimate()
            )
        )


if __name__ == "__main__":
    unittest.main()
