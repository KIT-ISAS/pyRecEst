"""Multi-target tracking example with a linear/Gaussian multi-Bernoulli tracker.

The scenario contains two one-dimensional targets with constant-velocity
motion, position-only detections, one missed detection, and a few clutter
measurements.  The tracker keeps persistent labels for the two Bernoulli
components, performs global nearest-neighbor data association internally, and
prints the extracted labeled state estimates after each update.
"""

# pylint: disable=no-name-in-module,no-member

from pyrecest.backend import array, diag, get_backend_name
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters import BernoulliComponent, MultiBernoulliTracker


def make_tracker():
    """Create a two-track multi-Bernoulli tracker for the example scenario."""
    initial_covariance = diag(array([0.5, 0.25]))
    initial_components = [
        BernoulliComponent(
            0.95,
            GaussianDistribution(array([0.0, 1.0]), initial_covariance),
            label="target-1",
        ),
        BernoulliComponent(
            0.95,
            GaussianDistribution(array([10.0, -0.8]), initial_covariance),
            label="target-2",
        ),
    ]

    tracker_param = {
        "survival_probability": 0.99,
        "detection_probability": 0.92,
        "clutter_intensity": 0.02,
        "gating_probability": 0.999,
        "gating_distance_threshold": None,
        "pruning_threshold": 0.05,
        "maximum_number_of_components": 10,
        "birth_existence_probability": 0.7,
        "birth_covariance": None,
        "measurement_to_state_matrix": None,
    }

    return MultiBernoulliTracker(
        initial_components,
        tracker_param=tracker_param,
        log_prior_estimates=False,
        log_posterior_estimates=False,
    )


def run_tracker():
    """Run prediction and update steps and return labeled estimates."""
    dt = 1.0
    system_matrix = array([[1.0, dt], [0.0, 1.0]])
    system_noise_covariance = diag(array([0.03, 0.01]))
    measurement_matrix = array([[1.0, 0.0]])
    measurement_noise_covariance = array([[0.16]])

    measurements_by_step = [
        array([[1.10, 9.15, 5.00]]),
        array([[2.05, 8.35]]),
        array([[3.20]]),
        array([[4.10, 6.95, 11.50]]),
        array([[5.05, 6.25]]),
    ]

    tracker = make_tracker()
    history = []

    for step, measurements in enumerate(measurements_by_step, start=1):
        tracker.predict_linear(system_matrix, system_noise_covariance)
        tracker.update_linear(
            measurements,
            measurement_matrix,
            measurement_noise_covariance,
        )

        labels, estimates = tracker.get_labeled_point_estimate(number_of_targets=2)
        components_by_label = tracker.get_labeled_components(copy_components=False)
        step_estimates = []
        for column, label in enumerate(labels):
            step_estimates.append(
                {
                    "label": label,
                    "position": float(estimates[0, column]),
                    "velocity": float(estimates[1, column]),
                    "existence_probability": components_by_label[
                        label
                    ].existence_probability,
                }
            )
        history.append((step, measurements, step_estimates))

    return history


def main():
    """Print a compact table of the labeled MTT estimates."""
    if get_backend_name() != "numpy":
        raise RuntimeError(
            "This multi-Bernoulli tracking example currently requires the "
            "numpy backend."
        )

    print("step measurements label     position velocity existence")
    for step, measurements, estimates in run_tracker():
        measurements_as_text = ",".join(
            f"{float(measurements[0, index]):.2f}"
            for index in range(measurements.shape[1])
        )
        for row_index, estimate in enumerate(
            sorted(estimates, key=lambda row: row["label"])
        ):
            measurement_cell = measurements_as_text if row_index == 0 else ""
            print(
                f"{step:>4} {measurement_cell:<16} "
                f"{estimate['label']:<8} "
                f"{estimate['position']:>8.3f} "
                f"{estimate['velocity']:>8.3f} "
                f"{estimate['existence_probability']:>9.3f}"
            )


if __name__ == "__main__":
    main()
