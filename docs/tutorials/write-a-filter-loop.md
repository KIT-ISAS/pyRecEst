# Write A Filter Loop

Most recursive filters alternate between prediction and update:

1. Predict the next state using the system model.
2. Update the state with the current measurement.
3. Read the posterior estimate or keep the filter state for the next step.

The following example runs a one-dimensional constant-velocity Kalman filter.
The state is `[position, velocity]`, and each measurement observes position.

```python
from pyrecest.backend import array, diag
from pyrecest.filters import KalmanFilter


dt = 1.0
system_matrix = array([[1.0, dt], [0.0, 1.0]])
measurement_matrix = array([[1.0, 0.0]])
system_noise_cov = diag(array([0.05, 0.01]))
measurement_noise_cov = array([[0.25]])
measurements = [0.9, 2.0, 3.1, 3.9, 5.2]

kalman_filter = KalmanFilter(
    (array([0.0, 1.0]), diag(array([1.0, 1.0])))
)

for step, measurement in enumerate(measurements, start=1):
    kalman_filter.predict_linear(system_matrix, system_noise_cov)
    kalman_filter.update_linear(
        array([measurement]), measurement_matrix, measurement_noise_cov
    )

    position, velocity = kalman_filter.get_point_estimate()
    print(
        f"{step}: position={float(position):.3f}, "
        f"velocity={float(velocity):.3f}"
    )

print("final covariance:")
print(kalman_filter.filter_state.C)
```

Run the same task as an executable example with:

```bash
python examples/basic/kalman_filter.py
```

## What To Notice

- `KalmanFilter` accepts either a `GaussianDistribution` or a `(mean,
  covariance)` tuple as its initial state.
- `predict_linear()` applies the system matrix and system-noise covariance.
- `update_linear()` applies the measurement matrix and measurement-noise
  covariance.
- `get_point_estimate()` returns the current posterior mean.
- `filter_state` returns a `GaussianDistribution` snapshot of the current
  posterior state.
