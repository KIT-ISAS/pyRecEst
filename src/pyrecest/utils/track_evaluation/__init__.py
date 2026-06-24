"""Compatibility wrapper for track-matrix evaluation helpers.

This wrapper keeps the existing implementation in ``track_evaluation.py`` while
patching track-matrix normalization so boolean-like cells are treated as missing
rather than as observation ids 0 and 1.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np

_SOURCE_PATH = Path(__file__).resolve().parent.parent / "track_evaluation.py"
_SPEC = importlib.util.spec_from_file_location(f"{__name__}._source", _SOURCE_PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - importlib guard
    raise ImportError(f"could not load track evaluation source from {_SOURCE_PATH}")
_SOURCE_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_SOURCE_MODULE)

_MISSING = _SOURCE_MODULE._MISSING  # pylint: disable=protected-access
_ORIGINAL_OPTIONAL_INT_CANDIDATE = _SOURCE_MODULE._optional_int_candidate  # pylint: disable=protected-access


def _optional_int_candidate(value: Any) -> Any:
    while isinstance(value, np.ndarray):
        if value.ndim != 0:
            return _MISSING
        value = value.item()
    if isinstance(value, (bool, np.bool_)):
        return _MISSING
    return _ORIGINAL_OPTIONAL_INT_CANDIDATE(value)


_SOURCE_MODULE._optional_int_candidate = _optional_int_candidate  # pylint: disable=protected-access

for _name in _SOURCE_MODULE.__all__:
    globals()[_name] = getattr(_SOURCE_MODULE, _name)

__all__ = _SOURCE_MODULE.__all__
