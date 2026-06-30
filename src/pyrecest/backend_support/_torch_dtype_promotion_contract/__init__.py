"""PyTorch backend logical-helper compatibility patch."""

from __future__ import annotations

import importlib.util
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
    _patch_pytorch_creation_shape_bool_contract(raw_pytorch, backend, torch)


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


def _shape_contains_boolean_dimension(shape, torch_module):
    """Return whether ``shape`` contains a boolean dimension value."""
    try:
        import numpy as np  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - NumPy is a core dependency
        np = None

    boolean_types = (bool,) if np is None else (bool, np.bool_)
    if isinstance(shape, boolean_types):
        return True
    if torch_module.is_tensor(shape):
        return shape.dtype == torch_module.bool
    if np is None:
        return False

    try:
        shape_values = np.asarray(shape, dtype=object).reshape(-1)
    except (TypeError, ValueError, RuntimeError):
        return False
    return any(isinstance(one_dimension, boolean_types) for one_dimension in shape_values)


def _validate_creation_shape_not_boolean(shape, torch_module) -> None:
    """Reject boolean shape values before the base PyTorch creation wrapper runs."""
    if _shape_contains_boolean_dimension(shape, torch_module):
        raise TypeError("shape dimensions must be integers, not booleans")


def _patch_pytorch_creation_shape_bool_contract(raw_pytorch, backend, torch) -> None:
    """Keep PyTorch creation helpers from treating booleans as dimensions."""
    helper_specs = (
        ("empty", False),
        ("zeros", False),
        ("ones", False),
        ("full", True),
    )
    active_pytorch_backend = getattr(backend, "__backend_name__", None) == "pytorch"
    if all(
        getattr(getattr(raw_pytorch, helper_name, None), "_pyrecest_rejects_bool_shape", False)
        for helper_name, _ in helper_specs
    ):
        if active_pytorch_backend:
            for helper_name, _ in helper_specs:
                setattr(backend, helper_name, getattr(raw_pytorch, helper_name))
        return

    def _wrap_creation_helper(helper_name, *, has_fill_value=False):
        original_helper = getattr(raw_pytorch, helper_name)
        if getattr(original_helper, "_pyrecest_rejects_bool_shape", False):
            return original_helper

        if has_fill_value:

            def creation_helper(shape, fill_value, *args, **kwargs):
                _validate_creation_shape_not_boolean(shape, torch)
                return original_helper(shape, fill_value, *args, **kwargs)

        else:

            def creation_helper(shape, *args, **kwargs):
                _validate_creation_shape_not_boolean(shape, torch)
                return original_helper(shape, *args, **kwargs)

        creation_helper.__name__ = getattr(original_helper, "__name__", helper_name)
        creation_helper.__doc__ = getattr(original_helper, "__doc__", None)
        creation_helper._pyrecest_rejects_bool_shape = True
        return creation_helper

    for helper_name, has_fill_value in helper_specs:
        helper = _wrap_creation_helper(helper_name, has_fill_value=has_fill_value)
        setattr(raw_pytorch, helper_name, helper)
        if active_pytorch_backend:
            setattr(backend, helper_name, helper)


__all__ = ["patch_pytorch_dtype_promotion_contract"]