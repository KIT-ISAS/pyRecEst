"""Kalman filter example using reusable linear-Gaussian model objects."""

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, diag
from pyrecest.filters import KalmanFilter
from pyrecest.models import (
    LinearGaussianMeasurementModel,
    LinearGaussianTransitionModel,
)


def run_filter():
    """Run a constant-velocity Kalman filter using model objects."""
    dt = 1.0
    transition_model = LinearGaussianTransitionModel(
        system_matrix=array([[1.0, dt], [0.0, 1.0]]),
        system_noise_cov=diag(array([0.05, 0.01])),
    )
    measurement_model = LinearGaussianMeasurementModel(
        measurement_matrix=array([[1.0, 0.0]]),
        meas_noise=array([[0.25]]),
    )

    measurements = [0.9, 2.0, 3.1, 3.9, 5.2]
    kalman_filter = KalmanFilter((array([0.0, 1.0]), diag(array([1.0, 1.0]))))
    estimates = []

    for measurement in measurements:
        kalman_filter.predict_model(transition_model)
        kalman_filter.update_model(measurement_model, array([measurement]))
        estimates.append(kalman_filter.get_point_estimate())

    return measurements, estimates, kalman_filter.filter_state.C


def main():
    """Print the posterior estimates produced by the filter loop."""
    measurements, estimates, final_covariance = run_filter()

    print("step  measurement  position  velocity")
    for step, (measurement, estimate) in enumerate(
        zip(measurements, estimates), start=1
    ):
        position, velocity = estimate
        print(
            f"{step:>4}  {measurement:>11.2f}  "
            f"{float(position):>8.3f}  {float(velocity):>8.3f}"
        )

    print("\nFinal covariance:")
    print(final_covariance)


if __name__ == "__main__":
    main()
