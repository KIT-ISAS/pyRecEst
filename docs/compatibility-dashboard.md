# Compatibility Dashboard

This page is a checked-in snapshot of the compatibility dimensions that should
also be generated in CI with `scripts/generate_compatibility_dashboard.py`.

## Python Versions

PyRecEst declares support for Python `>=3.11,<3.15`.

## Backend Support

Run the following command to generate the current public backend API matrix:

```bash
pyrecest backends --format markdown
```

## Scenario Coverage

At minimum, keep one linear-Gaussian scenario executable as a release smoke test.
As the scenario zoo grows, this page should list which scenarios run under each
backend and Python version.
