import numpy.testing as npt

from pyrecest.backend import array, eye
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters import GlobalNearestNeighbor, KalmanFilter


def _make_tracker(initial_means, covariance_scale=1.0, association_param=None):
    tracker = GlobalNearestNeighbor(association_param=association_param)
    tracker.filter_state = [
        KalmanFilter(GaussianDistribution(array(mu), covariance_scale * eye(len(mu))))
        for mu in initial_means
    ]
    return tracker


def test_update_linear_accepts_precomputed_cost_matrix_and_overrides_geometry():
    tracker_custom = _make_tracker(
        [[0.0, 0.0], [10.0, 0.0]],
        association_param={"gating_distance_threshold": 100.0, "max_new_tracks": 1},
    )
    tracker_default = _make_tracker(
        [[0.0, 0.0], [10.0, 0.0]],
        association_param={"gating_distance_threshold": 100.0, "max_new_tracks": 1},
    )

    measurements = array([[1.0, 9.0], [0.0, 0.0]])
    measurement_matrix = eye(2)
    measurement_cov = 0.01 * eye(2)

    # Force the opposite assignment from the geometric default.
    appearance_costs = array([[100.0, 0.0], [0.0, 100.0]])

    tracker_default.update_linear(measurements, measurement_matrix, measurement_cov)
    tracker_custom.update_linear(
        measurements,
        measurement_matrix,
        measurement_cov,
        pairwise_cost_matrix=appearance_costs,
    )

    default_estimates = tracker_default.get_point_estimate()
    custom_estimates = tracker_custom.get_point_estimate()

    # Default GNN should stay near the geometrically nearest measurements.
    assert default_estimates[0, 0] < 2.0
    assert default_estimates[0, 1] > 8.0

    # The precomputed cost matrix should intentionally swap the two matches.
    assert custom_estimates[0, 0] > 8.0
    assert custom_estimates[0, 1] < 2.0


def test_update_linear_skips_dummy_assignment_from_precomputed_cost_matrix():
    tracker = _make_tracker(
        [[0.0, 0.0]],
        association_param={"gating_distance_threshold": 10.0, "max_new_tracks": 2},
    )

    prior_estimate = tracker.get_point_estimate().copy()
    measurements = array([[100.0], [0.0]])
    impossible_costs = array([[float("inf")]])

    tracker.update_linear(
        measurements,
        eye(2),
        eye(2),
        pairwise_cost_matrix=impossible_costs,
    )

    npt.assert_allclose(tracker.get_point_estimate(), prior_estimate)
