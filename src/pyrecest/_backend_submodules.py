"""Utilities for exposing virtual backend submodules."""

from __future__ import annotations

import sys
from types import ModuleType

from pyrecest._backend import BACKEND_ATTRIBUTES


def register_backend_submodules(backend: ModuleType | None = None) -> None:
    """Register virtual backend submodules for standard import statements."""
    if backend is None:
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel

    backend.__path__ = getattr(backend, "__path__", [])
    backend_spec = getattr(backend, "__spec__", None)
    if backend_spec is not None:
        backend_spec.submodule_search_locations = (
            getattr(backend_spec, "submodule_search_locations", None) or []
        )

    for submodule_name in BACKEND_ATTRIBUTES:
        if not submodule_name:
            continue
        sys.modules.setdefault(
            f"{backend.__name__}.{submodule_name}",
            getattr(backend, submodule_name),
        )
