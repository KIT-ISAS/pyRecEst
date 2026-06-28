"""PyTorch dtype helper package with complete dtype alias coverage."""

from __future__ import annotations

import importlib.util as _importlib_util
import sys as _sys
from pathlib import Path as _Path

import torch as _torch

_LEGACY_MODULE_PATH = _Path(__file__).resolve().parent.parent / "_dtype.py"
_LEGACY_MODULE_NAME = "pyrecest._backend.pytorch._dtype_legacy"
_SPEC = _importlib_util.spec_from_file_location(
    _LEGACY_MODULE_NAME,
    _LEGACY_MODULE_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - importlib guard
    raise ImportError(f"Cannot load PyTorch dtype helpers from {_LEGACY_MODULE_PATH}")

_legacy_module = _importlib_util.module_from_spec(_SPEC)
_sys.modules[_LEGACY_MODULE_NAME] = _legacy_module
_SPEC.loader.exec_module(_legacy_module)

_legacy_module.MAP_DTYPE.update(
    {
        "bool": _torch.bool,
        "uint8": _torch.uint8,
        "int8": _torch.int8,
        "int16": _torch.int16,
        "int32": _torch.int32,
        "int64": _torch.int64,
        "float16": _torch.float16,
    }
)

globals().update(
    {
        name: getattr(_legacy_module, name)
        for name in dir(_legacy_module)
        if not name.startswith("__")
    }
)
