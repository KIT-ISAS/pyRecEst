# API Stability And Deprecation Policy

PyRecEst can support both stable research workflows and fast-moving experimental
ideas by separating public APIs from experimental APIs.

## Stability Categories

| Category         | Meaning                                                                                              |
|------------------|------------------------------------------------------------------------------------------------------|
| Stable           | Public API covered by tests and deprecation policy.                                                  |
| Experimental     | API may change without the full deprecation cycle. Prefer `pyrecest.experimental` for these objects. |
| Deprecated       | API still exists but emits `DeprecationWarning` and has a planned removal version.                   |
| Backend-specific | API is stable only for the backends listed in the backend API matrix.                                |

Tracked user-facing APIs are listed in the [public API registry](public-api-registry.md).
Keep that registry, backend capability metadata, and deprecation tests in sync
when API status changes.

## Deprecations

Use `pyrecest.deprecation.deprecated` for public API transitions:

```python
from pyrecest.deprecation import deprecated

@deprecated(since="2.3.0", remove_in="3.0.0", replacement="new_function")
def old_function():
    ...
```

Recommended cadence:

1. introduce the replacement and warning in a minor release;
2. keep the warning for at least one additional minor release;
3. remove only in a major release unless the API was explicitly experimental.

## Executable Stability Checks

Public API changes should be visible in tests or generated metadata. For new
stable, backend-specific, or deprecated APIs, update the relevant rows in the
backend API matrix and add one of the following:

- a focused behavior test for the stable contract;
- a backend-contract test that verifies supported, partial, and unsupported
  backends behave as documented;
- a deprecation test that asserts `DeprecationWarning` is emitted and that the
  replacement is named in the warning message;
- an explicit `experimental` documentation note when the API is not yet covered
  by the full deprecation cycle.

Treat undocumented package-level exports as accidental until they are covered by
this policy or moved under an experimental namespace.
