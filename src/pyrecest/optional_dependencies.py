"""Helpers for lazy optional dependencies."""

from __future__ import annotations

import importlib
from types import ModuleType

from pyrecest.exceptions import OptionalDependencyError


def require_optional_dependency(
    package: str, extra: str, *, feature: str | None = None
) -> ModuleType:
    """Import an optional dependency or raise a standardized error.

    Parameters
    ----------
    package:
        Import name, for example ``"matplotlib"`` or ``"healpy"``.
    extra:
        PyRecEst extra that installs the dependency.
    feature:
        Optional user-facing feature name for the error message.
    """
    try:
        return importlib.import_module(package)
    except (
        ImportError
    ) as exc:  # pragma: no cover - exercised through tests with a missing sentinel package
        subject = f" for {feature}" if feature else ""
        raise OptionalDependencyError(
            f"Optional dependency {package!r} is required{subject}. "
            f"Install it with `python -m pip install 'pyrecest[{extra}]'`."
        ) from exc


__all__ = ["require_optional_dependency"]
