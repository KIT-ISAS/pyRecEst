"""Utilities for exposing virtual backend submodules."""

from __future__ import annotations

import sys
from functools import wraps
from types import ModuleType

from pyrecest._backend import BACKEND_ATTRIBUTES


def _copy_result_to_out(result, out):
    """Copy ``result`` into a backend ``out`` object and return that object."""
    copy_ = getattr(out, "copy_", None)
    if copy_ is not None:
        copy_(result)
        return out
    try:
        out[...] = result
    except TypeError:
        at = getattr(out, "at", None)
        if at is None:
            raise
        return at[...].set(result)
    return out


def _cumulative_with_out(cumulative):
    """Return a cumulative helper accepting NumPy's ``out`` keyword."""

    @wraps(cumulative)
    def wrapped_cumulative(x, axis=None, dtype=None, out=None):
        result = cumulative(x, axis=axis, dtype=dtype)
        if out is not None:
            return _copy_result_to_out(result, out)
        return result

    wrapped_cumulative._pyrecest_out_contract = True
    return wrapped_cumulative


def _adapt_cumulative_out_contract(backend: ModuleType) -> None:
    """Adapt PyTorch cumulative helpers to the public NumPy-style contract."""
    if getattr(backend, "__backend_name__", None) != "pytorch":
        return
    for attribute_name in ("cumsum", "cumprod"):
        cumulative = getattr(backend, attribute_name, None)
        if cumulative is None or getattr(cumulative, "_pyrecest_out_contract", False):
            continue
        setattr(backend, attribute_name, _cumulative_with_out(cumulative))


def register_backend_submodules(backend: ModuleType | None = None) -> None:
    """Register virtual backend submodules for standard import statements."""
    if backend is None:
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel

    _adapt_cumulative_out_contract(backend)

    backend.__path__ = getattr(backend, "__path__", [])
    backend_spec = getattr(backend, "__spec__", None)
    if backend_spec is not None:
        backend_spec.submodule_search_locations = (
            getattr(backend_spec, "submodule_search_locations", None) or []
        )

    for submodule_name in BACKEND_ATTRIBUTES:
        if not submodule_name:
            continue
        sys.modules[f"{backend.__name__}.{submodule_name}"] = getattr(
            backend, submodule_name
        )
