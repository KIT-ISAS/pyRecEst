# PyRecEst Documentation

PyRecEst is a Python library for recursive Bayesian estimation on Euclidean
spaces and manifolds. The library includes distributions, filters, trackers,
smoothers, samplers, evaluation helpers, and a backend abstraction for NumPy,
PyTorch, and JAX.

These Markdown pages are the starting point for project documentation. They are
kept lightweight so they can be read directly on GitHub while the project grows
toward a generated documentation site.

## Start Here

- [Getting started](getting-started.md): install PyRecEst, run the examples,
  choose a backend, and set up a development checkout.
- [API overview](api-overview.md): understand the main packages and where common
  public classes live.
- [Backend compatibility](backend-compatibility.md): choose between NumPy,
  PyTorch, and JAX and understand known support gaps.
- [Shapes and conventions](conventions.md): learn the expected state,
  measurement, covariance, batch, and manifold-coordinate shapes.
- [Examples](../examples/README.md): browse executable scripts that demonstrate
  basic workflows.

## Current Documentation Scope

The README gives the shortest install and Kalman filter quickstart. The pages in
this directory add more orientation, but the tests are still the most complete
source of usage coverage for many advanced distributions, filters, trackers,
smoothers, samplers, and evaluation helpers.

Good next documentation additions would be:

- generated API reference from public docstrings;
- task-focused tutorials for distributions, filters, smoothers, tracking, and
  evaluation;
- deeper convention notes for grids, state-space subdivisions, and advanced
  tracker outputs;
- API-specific backend support tables for advanced distributions, trackers,
  evaluators, and utilities.
