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

## Before Opening a Pull Request

- Add or update tests for changed behavior.
- Update `docs/backend-compatibility.md` and
  `src/pyrecest/_backend/capabilities.py` when backend support changes.
- Run `python scripts/check_release_consistency.py --local-only` after changing
  release metadata, citation metadata, or package metadata.
- Keep examples executable from the repository root.

## Release Metadata

Version, citation, GitHub release, and PyPI state should agree for public
releases. The manual `Check release consistency` workflow can compare the local
version with GitHub and PyPI after a release is published.
