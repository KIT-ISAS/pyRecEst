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

## Public Symbol Registry

Public package namespaces should keep their exported symbols explicit. For lazy
namespaces, the export map is the registry and `__all__` is generated from it.
Every new public symbol should be classified as one of:

| Class               | Requirement                                                                                          |
|---------------------|------------------------------------------------------------------------------------------------------|
| Canonical           | Preferred spelling used in docs, examples, and release notes.                                        |
| Compatibility alias | Kept for old code or MATLAB/libDirectional naming compatibility; should point to a canonical symbol. |
| Experimental        | Allowed to move more quickly and documented as such.                                                 |

Avoid adding a new alias unless it solves a concrete compatibility problem. If
an alias is no longer needed, mark it with `pyrecest.deprecation.deprecated`
before removal.

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

## Runtime Metadata

Public APIs can expose stability metadata through `pyrecest.stability`:

```python
from pyrecest.stability import get_public_api_status

status = get_public_api_status("KalmanFilter")
```

Decorate new public functions, classes, and adapters with `@stability(...)` when
adding them to the package-level namespace. The decorator is intentionally small:
it stores metadata on the object and leaves import behavior unchanged.

## Public Surface Convention

- Objects imported from package-level modules such as `pyrecest.filters`,
  `pyrecest.distributions`, `pyrecest.models`, and `pyrecest.sampling` are public
  unless explicitly documented as experimental.
- Modules and attributes prefixed with `_` remain internal implementation detail.
- Backend-specific APIs are public only for the backends declared in the backend
  API matrix.
