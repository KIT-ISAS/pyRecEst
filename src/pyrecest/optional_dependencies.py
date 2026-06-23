"""Helpers for lazy optional dependencies."""

from __future__ import annotations

import importlib
from types import ModuleType

from pyrecest.exceptions import OptionalDependencyError


def _is_missing_requested_package(exc: ModuleNotFoundError, package: str) -> bool:
    missing_name = exc.name
    if missing_name is None:
        return False
    return missing_name == package or package.startswith(f"{missing_name}.")


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
    except ModuleNotFoundError as exc:
        if not _is_missing_requested_package(exc, package):
            raise
        subject = f" for {feature}" if feature else ""
        raise OptionalDependencyError(
            f"Optional dependency {package!r} is required{subject}. "
            f"Install it with `python -m pip install 'pyrecest[{extra}]'`."
        ) from exc


__all__ = ["require_optional_dependency"]
