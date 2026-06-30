"""PyTorch backend logical-helper compatibility patch."""

from __future__ import annotations

import importlib.util
from operator import index as _operator_index
from pathlib import Path


def _load_base_contract_module():
    module_path = Path(__file__).resolve().parent.parent / "_torch_dtype_promotion_contract.py"
    spec = importlib.util.spec_from_file_location(
        "_pyrecest_torch_dtype_promotion_contract_base",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load PyTorch dtype contract module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_BASE_CONTRACT = _load_base_contract_module()


def patch_pytorch_dtype_promotion_contract() -> None:
    """Apply the base PyTorch contract patch plus logical-helper device fixes."""
    _BASE_CONTRACT.patch_pytorch_dtype_promotion_contract()
    try:
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend import failed earlier
        return

    _patch_pytorch_logical_device_contract(raw_pytorch, backend, torch)
    _patch_pytorch_diagonal_axis_contract(raw_pytorch, backend, torch)


def _preferred_pytorch_device(torch_module, *values):
    """Return a non-CPU tensor device when mixed-device operands are present."""
    for value in values:
        if torch_module.is_tensor(value) and value.device.type != "cpu":
            return value.device
    for value in values:
        if torch_module.is_tensor(value):
            return value.device
    return None


def _as_pytorch_tensor_on_device(value, torch_module, *, device, dtype=None):
    """Return ``value`` as a tensor on ``device``."""
    if torch_module.is_tensor(value):
        if device is not None and value.device != device:
            value = value.to(device=device)
        if dtype is not None and value.dtype != dtype:
            value = value.to(dtype=dtype)
        return value
    return torch_module.as_tensor(value, dtype=dtype, device=device)


def _normalize_pytorch_diagonal_integer(value, torch_module, name):
    """Return a NumPy-style scalar integer argument for ``diagonal``."""
    if torch_module.is_tensor(value):
        if (
            value.ndim != 0
            or value.dtype == torch_module.bool
            or value.dtype.is_floating_point
            or value.dtype.is_complex
        ):
            raise TypeError(f"{name} must be an integer")
        return _operator_index(value.item())
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        return _operator_index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc


def _patch_pytorch_diagonal_axis_contract(raw_pytorch, backend, torch) -> None:
    """Make PyTorch diagonal accept NumPy-style scalar integer indices."""
    original_diagonal = getattr(raw_pytorch, "diagonal", None)
    if original_diagonal is None:
        return
    if getattr(original_diagonal, "_pyrecest_axis_contract", False):
        if getattr(backend, "__backend_name__", None) == "pytorch":
            backend.diagonal = original_diagonal
        return

    def diagonal(x, offset=0, axis1=0, axis2=1):
        return original_diagonal(
            x,
            offset=_normalize_pytorch_diagonal_integer(offset, torch, "offset"),
            axis1=_normalize_pytorch_diagonal_integer(axis1, torch, "axis1"),
            axis2=_normalize_pytorch_diagonal_integer(axis2, torch, "axis2"),
        )

    diagonal.__name__ = getattr(original_diagonal, "__name__", "diagonal")
    diagonal.__doc__ = getattr(original_diagonal, "__doc__", None)
    diagonal._pyrecest_axis_contract = True
    raw_pytorch.diagonal = diagonal
    if getattr(backend, "__backend_name__", None) == "pytorch":
        backend.diagonal = diagonal


def _patch_pytorch_logical_device_contract(raw_pytorch, backend, torch) -> None:
    """Keep logical helpers on an existing non-CPU tensor device."""
    helper_names = ("logical_and", "where")
    if all(
        getattr(
            getattr(raw_pytorch, helper_name, None),
            "_pyrecest_device_contract",
            False,
        )
        for helper_name in helper_names
    ):
        if getattr(backend, "__backend_name__", None) == "pytorch":
            for helper_name in helper_names:
                setattr(backend, helper_name, getattr(raw_pytorch, helper_name))
        return

    original_logical_and = raw_pytorch.logical_and
    original_where = raw_pytorch.where

    def logical_and(x, y):
        device = _preferred_pytorch_device(torch, x, y)
        return torch.logical_and(
            _as_pytorch_tensor_on_device(x, torch, device=device),
            _as_pytorch_tensor_on_device(y, torch, device=device),
        )

    def where(condition, x=None, y=None):
        device = _preferred_pytorch_device(torch, condition, x, y)
        condition = _as_pytorch_tensor_on_device(
            condition,
            torch,
            device=device,
            dtype=torch.bool,
        )

        if x is None and y is None:
            return torch.where(condition)
        if x is None or y is None:
            raise ValueError("either both or neither of x and y should be given")

        x = _as_pytorch_tensor_on_device(x, torch, device=device)
        y = _as_pytorch_tensor_on_device(y, torch, device=device)
        result_dtype = torch.result_type(x, y)
        return torch.where(
            condition,
            x.to(dtype=result_dtype),
            y.to(dtype=result_dtype),
        )

    logical_and.__name__ = getattr(original_logical_and, "__name__", "logical_and")
    logical_and.__doc__ = getattr(original_logical_and, "__doc__", None)
    logical_and._pyrecest_device_contract = True
    where.__name__ = getattr(original_where, "__name__", "where")
    where.__doc__ = getattr(original_where, "__doc__", None)
    where._pyrecest_device_contract = True

    raw_pytorch.logical_and = logical_and
    raw_pytorch.where = where
    if getattr(backend, "__backend_name__", None) == "pytorch":
        backend.logical_and = logical_and
        backend.where = where


__all__ = ["patch_pytorch_dtype_promotion_contract"]
