# Run A Tracker

Trackers maintain target state estimates over time. This tutorial initializes a
small labeled multi-Bernoulli tracker with two one-dimensional targets, then
processes two cluttered measurement sets.

This example currently requires the NumPy backend.

```python
from pyrecest.backend import array, diag, get_backend_name
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters import BernoulliComponent, MultiBernoulliTracker


if get_backend_name() != "numpy":
    raise RuntimeError("This tutorial currently requires the numpy backend.")

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

tracker = MultiBernoulliTracker(
    initial_components,
    tracker_param={
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
    },
    log_prior_estimates=False,
    log_posterior_estimates=False,
)

system_matrix = array([[1.0, 1.0], [0.0, 1.0]])
system_noise_covariance = diag(array([0.03, 0.01]))
measurement_matrix = array([[1.0, 0.0]])
measurement_noise_covariance = array([[0.16]])
measurements_by_step = [
    array([[1.10, 9.15, 5.00]]),
    array([[2.05, 8.35]]),
]

for step, measurements in enumerate(measurements_by_step, start=1):
    tracker.predict_linear(system_matrix, system_noise_covariance)
    tracker.update_linear(
        measurements,
        measurement_matrix,
        measurement_noise_covariance,
    )

    labels, estimates = tracker.get_labeled_point_estimate(number_of_targets=2)
    print(f"step {step}")
    for column, label in enumerate(labels):
        print(
            f"  {label}: position={float(estimates[0, column]):.3f}, "
            f"velocity={float(estimates[1, column]):.3f}"
        )
```

Run the longer executable example with:

```bash
python examples/basic/multi_target_tracking.py
```

## What To Notice

- Each `BernoulliComponent` has an existence probability, a state distribution,
  and a persistent label.
- The measurement array has shape `(measurement_dim, n_measurements)`.
- `predict_linear()` propagates every component through the linear motion model.
- `update_linear()` associates measurements, handles clutter according to the
  tracker parameters, and updates component states.
- `get_labeled_point_estimate()` returns labels and a state matrix whose columns
  correspond to labeled target estimates.
