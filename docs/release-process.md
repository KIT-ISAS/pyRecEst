# Release Process

Use release changes as small, reviewable PRs whenever possible.

## Recommended PR Split

| PR type            | Typical contents                                           |
|--------------------|------------------------------------------------------------|
| Packaging          | Version metadata, dependency ranges, wheel/sdist checks.   |
| Backend contract   | Capability matrix updates, backend-specific tests, docs.   |
| Numerical behavior | Validation helpers, algorithmic changes, invariant tests.  |
| Documentation      | Tutorials, examples, compatibility dashboard, release notes. |
| Benchmarks         | ASV or JSON benchmark updates.                             |

## Release Notes

Generate a first draft from commit subjects:

```bash
python scripts/generate_release_notes.py v2.2.1..HEAD --output RELEASE_NOTES.md
```

Then edit the draft to emphasize user-facing changes, backend-support changes,
deprecations, dependency changes, and known limitations.

## Post-release Checks

After publishing, verify that the GitHub tag, GitHub release, PyPI version, and
`pyproject.toml` version all agree. Then install the published wheel in a fresh
environment and run the README example plus the scenario zoo smoke tests.
