import unittest

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import allclose, array, diag, eye
from pyrecest.filters.distributed_kalman_filter import (
    DistributedKalmanFilter,
    LinearGaussianInformationContribution,
)
from pyrecest.filters.kalman_filter import KalmanFilter


class _LinearGaussianMeasurementModel:
    def __init__(self, measurement_matrix, meas_noise):
        self.measurement_matrix = measurement_matrix
        self.meas_noise = meas_noise


class DistributedKalmanFilterTest(unittest.TestCase):
    def test_initialization_replicates_single_state(self):
        dkf = DistributedKalmanFilter(
            (array([0.0, 1.0]), diag(array([1.0, 2.0]))),
            num_nodes=3,
        )

        self.assertEqual(dkf.num_nodes, 3)
        self.assertEqual(dkf.dim, 2)
        self.assertTrue(
            allclose(
                dkf.means,
                array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]),
            )
        )

    def test_distributed_linear_update_matches_sequential_kalman_update(self):
        initial_state = (array([0.0, 1.0]), diag(array([3.0, 4.0])))
        dkf = DistributedKalmanFilter(initial_state, num_nodes=3)
        centralized = KalmanFilter(initial_state)

        measurements = [array([1.0]), array([2.0]), array([0.5, -1.0])]
        measurement_matrices = [array([[1.0, 0.0]]), array([[0.0, 1.0]]), eye(2)]
        meas_noises = [array([[2.0]]), array([[3.0]]), diag(array([5.0, 6.0]))]

        for measurement, measurement_matrix, meas_noise in zip(
            measurements,
            measurement_matrices,
            meas_noises,
        ):
            centralized.update_linear(measurement, measurement_matrix, meas_noise)

        dkf.update_distributed_linear(measurements, measurement_matrices, meas_noises)

        for state in dkf.node_states:
            self.assertTrue(allclose(state.mu, centralized.filter_state.mu))
            self.assertTrue(allclose(state.C, centralized.filter_state.C))

    def test_distributed_linear_update_supports_different_node_priors(self):
        initial_states = [
            (array([0.0, 1.0]), diag(array([3.0, 4.0]))),
            (array([2.0, -1.0]), diag(array([5.0, 2.0]))),
        ]
        dkf = DistributedKalmanFilter(initial_states)

        measurements = [array([1.0]), array([2.0])]
        measurement_matrices = [array([[1.0, 0.0]]), array([[0.0, 1.0]])]
        meas_noises = [array([[2.0]]), array([[3.0]])]

        expected_filters = [KalmanFilter(state) for state in initial_states]
        for expected_filter in expected_filters:
            for measurement, measurement_matrix, meas_noise in zip(
                measurements,
                measurement_matrices,
                meas_noises,
            ):
                expected_filter.update_linear(measurement, measurement_matrix, meas_noise)

        dkf.update_distributed_linear(measurements, measurement_matrices, meas_noises)

        for state, expected_filter in zip(dkf.node_states, expected_filters):
            self.assertTrue(allclose(state.mu, expected_filter.filter_state.mu))
            self.assertTrue(allclose(state.C, expected_filter.filter_state.C))

    def test_update_from_information_contribution_can_update_selected_nodes(self):
        dkf = DistributedKalmanFilter((array([0.0]), array([[2.0]])), num_nodes=2)
        contribution = DistributedKalmanFilter.local_information_contribution(
            array([4.0]),
            array([[1.0]]),
            array([[2.0]]),
        )
        self.assertIsInstance(contribution, LinearGaussianInformationContribution)

        dkf.update_from_information_contribution(contribution, node_indices=[1])

        self.assertTrue(allclose(dkf.node_states[0].mu, array([0.0])))
        self.assertTrue(allclose(dkf.node_states[0].C, array([[2.0]])))
        self.assertTrue(allclose(dkf.node_states[1].mu, array([2.0])))
        self.assertTrue(allclose(dkf.node_states[1].C, array([[1.0]])))

    def test_update_distributed_model_matches_distributed_linear(self):
        initial_state = (array([0.0, 1.0]), diag(array([3.0, 4.0])))
        filter_linear = DistributedKalmanFilter(initial_state, num_nodes=2)
        filter_model = DistributedKalmanFilter(initial_state, num_nodes=2)

        measurements = [array([1.0]), array([2.0])]
        measurement_matrices = [array([[1.0, 0.0]]), array([[0.0, 1.0]])]
        meas_noises = [array([[2.0]]), array([[3.0]])]
        measurement_models = [
            _LinearGaussianMeasurementModel(measurement_matrix, meas_noise)
            for measurement_matrix, meas_noise in zip(measurement_matrices, meas_noises)
        ]

        filter_linear.update_distributed_linear(
            measurements,
            measurement_matrices,
            meas_noises,
        )
        filter_model.update_distributed_model(measurement_models, measurements)

        self.assertTrue(allclose(filter_linear.means, filter_model.means))
        self.assertTrue(allclose(filter_linear.covariances, filter_model.covariances))


if __name__ == "__main__":
    unittest.main()
