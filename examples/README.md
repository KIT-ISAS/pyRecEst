# Examples

This directory contains small executable examples that demonstrate common
PyRecEst workflows.

Run examples from the repository root after installing PyRecEst or after
installing a development checkout.

## Basic examples

### Gaussian multiplication

`basic/gaussian_multiplication.py` multiplies several two-dimensional Gaussian
distributions and checks the result against the closed-form information
representation.

Run it with:

```bash
python examples/basic/gaussian_multiplication.py
```

### Kalman filter

`basic/kalman_filter.py` runs a one-dimensional constant-velocity Kalman filter.
It demonstrates how to:

- import arrays through `pyrecest.backend`;
- initialize `KalmanFilter` with a mean and covariance;
- call `predict_linear` and `update_linear`; and
- read the current point estimate and covariance.

Run it from the repository root with:

```bash
python examples/basic/kalman_filter.py
```

### Multi-target tracking

`basic/multi_target_tracking.py` runs a small linear/Gaussian
multi-Bernoulli-tracker scenario with two labeled targets, missed detections,
and clutter measurements.

Run it with:

```bash
python examples/basic/multi_target_tracking.py
```

This example currently requires the NumPy backend.

### von Mises-Fisher multiplication

`basic/von_mises_fisher_multiplication.py` multiplies two von Mises-Fisher
distributions on the unit sphere and verifies the analytic product relation.

Run it with:

```bash
python examples/basic/von_mises_fisher_multiplication.py
```

## Backend selection

Select a non-default backend by setting `PYRECEST_BACKEND` before running the
script. For example, on a bash-compatible shell:

```bash
PYRECEST_BACKEND=pytorch python examples/basic/kalman_filter.py
```

Install the matching optional dependency extra before using a non-default
backend. See [backend compatibility](../docs/backend-compatibility.md) for
known backend-specific limitations.

## Notebooks

`notebooks/` is reserved for interactive examples.
