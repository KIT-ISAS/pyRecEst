## Summary

Describe the change and the user-facing behavior it affects.

## Checklist

- [ ] Tests were added or updated.
- [ ] Backend limitations were documented or added to `src/pyrecest/_backend/capabilities.py`.
- [ ] Shape, dtype, and measurement-set conventions were checked against `docs/conventions.md`.
- [ ] New user-facing failures use clear messages or shared exceptions from `pyrecest.exceptions`.
- [ ] Public examples and docs were updated when relevant.
- [ ] Performance-sensitive changes were checked against `benchmarks/basic_regressions.py` or a focused benchmark.
- [ ] `python scripts/check_release_consistency.py --local-only` passes when release metadata changed.
- [ ] The package smoke path still works: `python -m build`, `twine check dist/*`, install wheel, run a basic example.
