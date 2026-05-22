"""Use reliability-weighted measurements with the full SCGP tracker."""

from pyrecest.backend import array, eye
from pyrecest.filters import FullSCGPTracker


def make_tracker():
    n_base_points = 8
    return FullSCGPTracker(
        n_base_points,
        kinematic_state=array([0.0, 0.0, 0.0, 1.0, 0.0]),
        kinematic_covariance=1e-4 * eye(5),
        shape_state=array([1.0] * n_base_points),
        shape_covariance=0.05 * eye(n_base_points),
        measurement_noise=0.02 * eye(2),
        radial_noise_variance=0.01,
        extent_forgetting_rate=0.2,
        reference_extent=array([1.0] * n_base_points),
    )


tracker = make_tracker()
measurements = array(
    [
        [1.4, 0.1],
        [0.2, 1.2],
        [3.0, 3.0],
    ]
)

tracker.update(
    measurements,
    measurement_weights=array([1.0, 0.25, 0.0]),
    active_measurement_mask=array([True, True, False]),
)

print("active measurement indices:", tracker.last_active_measurement_indices)
print("measurement weights:", tracker.last_measurement_weights)
print("kinematic estimate:", tracker.get_point_estimate_kinematics())
