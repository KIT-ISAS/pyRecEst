#!/usr/bin/env python3
"""Small deterministic benchmark/regression scenarios for PyRecEst."""

from __future__ import annotations

from time import perf_counter

from pyrecest.backend import array, diag
from pyrecest.filters import KalmanFilter


def run_linear_kalman(iterations: int = 200):
    """Run a deterministic one-dimensional constant-velocity Kalman scenario."""
    dt = 1.0
    system_matrix = array([[1.0, dt], [0.0, 1.0]])
    measurement_matrix = array([[1.0, 0.0]])
    system_noise_cov = diag(array([0.05, 0.01]))
    measurement_noise_cov = array([[0.25]])

    kalman_filter = KalmanFilter((array([0.0, 1.0]), diag(array([1.0, 1.0]))))

    start = perf_counter()
    for step in range(iterations):
        measurement = array([float(step + 1)])
        kalman_filter.predict_linear(system_matrix, system_noise_cov)
        kalman_filter.update_linear(measurement, measurement_matrix, measurement_noise_cov)
    elapsed = perf_counter() - start

    return kalman_filter.get_point_estimate(), elapsed


def main() -> None:
    estimate, elapsed = run_linear_kalman()
    print(f"linear_kalman_estimate={estimate}")
    print(f"linear_kalman_elapsed_seconds={elapsed:.6f}")


if __name__ == "__main__":
    main()
