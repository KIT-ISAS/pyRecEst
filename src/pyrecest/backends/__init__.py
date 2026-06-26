"""Explicit backend access helpers.

The public ``pyrecest.backend`` facade is still selected through the
``PYRECEST_BACKEND`` environment variable for backwards compatibility.  This
module gives tests, tools, and advanced users a direct way to inspect or use a
concrete backend module without installing another import hook or changing the
process-wide active backend.
"""

from __future__ import annotations

import importlib
from types import ModuleType

from pyrecest.exceptions import OptionalDependencyError

SUPPORTED_BACKENDS: tuple[str, ...] = ("numpy", "pytorch", "jax", "autograd")
_OPTIONAL_BACKEND_EXTRAS: dict[str, str] = {
    "pytorch": "pytorch_support",
    "jax": "jax_support",
    "autograd": "jax_support",
}


def _normalize_backend_name(name: str) -> str:
    if not isinstance(name, str):
        raise ValueError("Backend name must be a string.")
    normalized = name.lower().strip()
    if normalized not in SUPPORTED_BACKENDS:
        supported = ", ".join(SUPPORTED_BACKENDS)
        raise ValueError(f"Unknown backend '{name}'. Supported backends: {supported}.")
    return normalized


def get_backend(name: str) -> ModuleType:
    """Return a concrete backend implementation module by name.

    Parameters
    ----------
    name:
        One of ``"numpy"``, ``"pytorch"``, ``"jax"``, or ``"autograd"``.
    """
    normalized = _normalize_backend_name(name)
    module_name = f"pyrecest._backend.{normalized}"
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            supported = ", ".join(SUPPORTED_BACKENDS)
            raise ValueError(
                f"Unknown backend '{name}'. Supported backends: {supported}."
            ) from exc
        extra = _OPTIONAL_BACKEND_EXTRAS.get(normalized)
        if extra is None:
            raise
        missing = f" {exc.name!r}" if exc.name else ""
        raise OptionalDependencyError(
            f"Backend '{normalized}' requires optional dependency{missing}. "
            f"Install it with `python -m pip install 'pyrecest[{extra}]'`."
        ) from exc


__all__ = ["SUPPORTED_BACKENDS", "get_backend"]
