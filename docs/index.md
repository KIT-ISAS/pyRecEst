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
- [Task tutorials](tutorials/index.md): work through common distribution,
  filtering, tracking, and evaluation tasks.
- [Shapes and conventions](conventions.md): learn the expected state,
  measurement, covariance, batch, and manifold-coordinate shapes.
- [Examples](examples.md): browse executable scripts that demonstrate basic
  workflows.
- [API reference](reference/index.md): generated reference pages for the main
  public packages.

## Current Documentation Scope

The README gives the shortest install and Kalman filter quickstart. The pages in
this directory add more orientation, but the tests are still the most complete
source of usage coverage for many advanced distributions, filters, trackers,
smoothers, samplers, and evaluation helpers.

Good next documentation additions would be:

- deeper convention notes for grids, state-space subdivisions, and advanced
  tracker outputs;
- backend compatibility notes for APIs that do not support every backend.
