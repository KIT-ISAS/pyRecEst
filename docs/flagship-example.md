# Flagship Example: Multi-target Tracking With Clutter

A flagship example should demonstrate why PyRecEst is useful beyond a minimal
Kalman-filter snippet. The recommended flagship is a compact multi-target
tracking scenario with missed detections and clutter.

## What It Should Show

1. A clear generative model for target motion and measurements.
2. A reusable transition model and measurement model.
3. Gating and association diagnostics.
4. Track lifecycle behavior across several time steps.
5. A plot or tabular summary of estimates, missed detections, and false tracks.
6. Backend notes explaining whether the workflow is NumPy-only, bridged, or
   portable.

## Acceptance Criteria

- The example runs from the repository root.
- It has a deterministic seed or golden expected output.
- It is small enough for CI smoke testing.
- It links to the backend API matrix and scenario-zoo entry.

This page is intentionally a specification first. The example implementation can
be expanded incrementally without changing the acceptance criteria.
