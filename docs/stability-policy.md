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

## Public Symbol Registry

Public package namespaces should keep their exported symbols explicit. For lazy
namespaces, the export map is the registry and `__all__` is generated from it.
Every new public symbol should be classified as one of:

| Class | Requirement |
|-------|-------------|
| Canonical | Preferred spelling used in docs, examples, and release notes. |
| Compatibility alias | Kept for old code or MATLAB/libDirectional naming compatibility; should point to a canonical symbol. |
| Experimental | Allowed to move more quickly and documented as such. |

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
