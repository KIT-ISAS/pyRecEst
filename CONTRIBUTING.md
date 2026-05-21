# Contributing to PyRecEst

Thank you for improving PyRecEst. Contributions are most useful when they keep
backend behavior, numerical conventions, and documentation aligned.

## Development Setup

```bash
poetry install --with dev --all-extras
python -m pytest
```

Run backend-specific checks by setting `PYRECEST_BACKEND` before importing
PyRecEst:

```bash
PYRECEST_BACKEND=numpy python -m pytest
PYRECEST_BACKEND=pytorch python -m pytest
PYRECEST_BACKEND=jax JAX_ENABLE_X64=True python -m pytest
```

## Nox Sessions

The repository also provides Nox sessions that mirror the main backend and
documentation workflows:

```bash
poetry run nox -s tests_numpy
poetry run nox -s tests_pytorch
poetry run nox -s tests_jax
poetry run nox -s docs
poetry run nox -s numerical_stress
poetry run nox -s benchmarks
```

The PyTorch and JAX sessions require the corresponding optional extras. JAX
sessions set `JAX_ENABLE_X64=True` to match the main CI configuration.

## Quality Gates

Use CI for the full matrix, but run focused local checks for the changed surface
area before opening a pull request:

```bash
python -m compileall -q src scripts tests
PYTHONPATH=src python scripts/render_backend_api_matrix.py --check docs/backend-api-matrix.md
PYTHONPATH=src python scripts/check_public_api_registry.py --check docs/public-api-registry.md
python scripts/check_minimal_imports.py
```

Backend selection is process-global and import-time only. Set
`PYRECEST_BACKEND` before importing `pyrecest`, and use
`pyrecest.assert_backend(...)` or `pyrecest.warn_if_backend_env_changed()` in
scripts where accidental backend changes would be confusing.

## Before Opening a Pull Request

- Add or update tests for changed behavior.
- Update `docs/backend-compatibility.md` and
  `src/pyrecest/_backend/capabilities.py` when backend support changes.
- Run `python scripts/check_release_consistency.py --local-only` after changing
  release metadata, citation metadata, or package metadata.
- Run `python scripts/render_backend_api_matrix.py --check docs/backend-api-matrix.md`
  after changing backend capability metadata.
- Run `python scripts/check_public_api_registry.py --check docs/public-api-registry.md`
  after adding, removing, stabilizing, deprecating, or reclassifying public APIs.
- Keep examples executable from the repository root.

## Adding or Changing Public APIs

When adding a distribution, filter, tracker, sampler, or utility that users are
expected to call directly, update the backend support metadata and the relevant
guide page in the same pull request.  Use the following decision order:

1. Decide whether the API is backend-portable, partially portable, or explicitly
   restricted to one backend.
2. Add or update a row in `src/pyrecest/_backend/capabilities.py`.
3. Add a focused backend contract test for any promised portable behavior.
4. Prefer `BackendNotSupportedError`, `ShapeError`, `DimensionMismatchError`, or
   `NumericalStabilityError` for new user-facing failures.
5. Add or update the matching row in `src/pyrecest/api_registry.py`.

## Release Metadata

Version, citation, GitHub release, and PyPI state should agree for public
releases. The manual `Check release consistency` workflow can compare the local
version with GitHub and PyPI after a release is published.
