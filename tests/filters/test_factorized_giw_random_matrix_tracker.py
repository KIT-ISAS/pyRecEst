import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, eye, linalg, mean
from pyrecest.filters import FactorizedGIWRandomMatrixTracker


class TestFactorizedGIWRandomMatrixTracker(unittest.TestCase):
    def test_extent_property_returns_inverse_wishart_mean(self):
        extent_dof = 14.0
        extent_scale = array([[16.0, 4.0], [4.0, 24.0]])
        tracker = FactorizedGIWRandomMatrixTracker(
            array([1.0, 2.0]),
            eye(2),
            extent_dof,
            extent_scale,
        )

        npt.assert_allclose(tracker.extent, extent_scale / 8.0)
        npt.assert_allclose(
            tracker.get_point_estimate_extent(flatten_matrix=True),
            array([2.0, 0.5, 0.5, 3.0]),
        )

    # pylint: disable=too-many-locals
    def test_update_matches_factorized_giw_matrix_form(self):
        prior_extent = array([[2.0, 0.4], [0.4, 1.2]])
        prior_dof = 12.0
        prior_scale = (prior_dof - 6.0) * prior_extent
        tracker = FactorizedGIWRandomMatrixTracker(
            array([0.2, -0.3]),
            array([[0.4, 0.05], [0.05, 0.3]]),
            prior_dof,
            prior_scale,
        )
        measurements = array([[1.4, 1.6, 1.2], [0.5, 0.2, 0.8]])
        meas_noise = array([[0.3, 0.05], [0.05, 0.25]])
        measurement_matrix = eye(2)

        y_mean = mean(measurements, axis=1, keepdims=True)
        ys_demean = measurements - y_mean
        measurement_scatter = ys_demean @ ys_demean.T
        predicted_measurement = measurement_matrix @ tracker.kinematic_state
        innovation = y_mean.flatten() - predicted_measurement

        Y = prior_extent + meas_noise
        S = measurement_matrix @ tracker.covariance @ measurement_matrix.T + Y / measurements.shape[1]
        gain = tracker.covariance @ linalg.solve(S, measurement_matrix).T
        expected_state = tracker.kinematic_state + gain @ innovation
        expected_covariance = tracker.covariance - gain @ S @ gain.T

        extent_sqrt = linalg.cholesky(prior_extent)
        innovation_sqrt = extent_sqrt @ linalg.solve(linalg.cholesky(S), innovation.reshape(-1, 1))
        expected_innovation_extent = innovation_sqrt @ innovation_sqrt.T
        extent_meas_sqrt = linalg.solve(linalg.cholesky(Y).T, extent_sqrt.T).T
        expected_scatter_extent = extent_meas_sqrt @ measurement_scatter @ extent_meas_sqrt.T
        expected_scale = prior_scale + expected_innovation_extent + expected_scatter_extent

        tracker.update(measurements, measurement_matrix, meas_noise)

        npt.assert_allclose(tracker.kinematic_state, expected_state)
        npt.assert_allclose(tracker.covariance, expected_covariance)
        npt.assert_allclose(tracker.extent_dof, prior_dof + measurements.shape[1])
        npt.assert_allclose(tracker.extent_scale, expected_scale)

    def test_predict_uses_factorized_giw_prediction(self):
        prior_dof = 20.0
        transition_dof = 100.0
        prior_scale = array([[28.0, 4.0], [4.0, 42.0]])
        tracker = FactorizedGIWRandomMatrixTracker(
            array([1.0, 0.5]),
            array([[0.6, 0.1], [0.1, 0.4]]),
            prior_dof,
            prior_scale,
            extent_transition_dof=transition_dof,
        )
        system_matrix = array([[1.0, 2.0], [0.0, 1.0]])
        process_noise = array([[0.2, 0.0], [0.0, 0.1]])

        tracker.predict(2.0, process_noise, system_matrix=system_matrix)

        extent_dimension = 2.0
        expected_dof = extent_dimension + 1.0 + (prior_dof - extent_dimension - 1.0) / (1.0 + (prior_dof - 2.0 * extent_dimension - 2.0) / transition_dof)
        expected_scale_factor = 1.0 / (1.0 + (prior_dof - extent_dimension - 1.0) / (transition_dof - extent_dimension - 1.0))

        npt.assert_allclose(tracker.kinematic_state, array([2.0, 0.5]))
        npt.assert_allclose(
            tracker.covariance,
            system_matrix @ array([[0.6, 0.1], [0.1, 0.4]]) @ system_matrix.T + process_noise,
        )
        npt.assert_allclose(tracker.extent_dof, expected_dof)
        npt.assert_allclose(tracker.extent_scale, expected_scale_factor * prior_scale)


if __name__ == "__main__":
    unittest.main()
