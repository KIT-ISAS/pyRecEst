"""Package wrapper for lazy optional-import helpers."""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any

from pyrecest.exceptions import OptionalDependencyError


def _validate_nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _validate_optional_feature(value: Any | None) -> str | None:
    if value is None:
        return None
    return _validate_nonempty_string(value, "feature")


def _is_missing_requested_package(exc: ModuleNotFoundError, package: str) -> bool:
    missing_name = exc.name
    if missing_name is None:
        return False
    return missing_name == package or package.startswith(f"{missing_name}.")


def require_optional_dependency(
    package: str, extra: str, *, feature: str | None = None
) -> ModuleType:
    """Import an optional package or raise a standardized error."""
    package = _validate_nonempty_string(package, "package")
    extra = _validate_nonempty_string(extra, "extra")
    feature = _validate_optional_feature(feature)

    try:
        return importlib.import_module(package)
    except ModuleNotFoundError as exc:
        if not _is_missing_requested_package(exc, package):
            raise
        subject = f" for {feature}" if feature else ""
        raise OptionalDependencyError(
            f"Optional package {package!r} is required{subject}. "
            f"Install the 'pyrecest[{extra}]' extra."
        ) from exc


__all__ = ["require_optional_dependency"]
