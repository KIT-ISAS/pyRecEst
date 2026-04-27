# Examples

This directory contains small executable examples that demonstrate common
PyRecEst workflows.

## Basic examples

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

Select a non-default backend by setting `PYRECEST_BACKEND` before running the
script, for example:

```bash
PYRECEST_BACKEND=pytorch python examples/basic/kalman_filter.py
```

Install the matching optional dependency extra before using a non-default
backend.

## Notebooks

`notebooks/` is reserved for interactive examples.
