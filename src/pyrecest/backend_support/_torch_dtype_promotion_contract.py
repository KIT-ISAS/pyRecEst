"""PyTorch backend compatibility helpers."""

from __future__ import annotations

from pyrecest.backend_support._torch_comparison_contract import (
    patch_pytorch_comparison_arraylike_contract,
)
from pyrecest.backend_support._torch_diff_contract import (
    patch_pytorch_diff_numpy_contract,
)
from pyrecest.backend_support._torch_pad_contract import (
    patch_pytorch_pad_constant_values_contract,
)
from pyrecest.backend_support._torch_repeat_contract import (
    patch_pytorch_repeat_numpy_contract,
)


def patch_pytorch_dtype_promotion_contract() -> None:
    """Make PyTorch backend helpers use PyRecEst compatibility contracts."""
    try:
        import numpy as np  # pylint: disable=import-outside-toplevel
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend import failed earlier
        return

    patch_pytorch_repeat_numpy_contract(raw_pytorch, backend, torch, np)
    patch_pytorch_diff_numpy_contract(raw_pytorch, backend, torch)
    patch_pytorch_pad_constant_values_contract(raw_pytorch, backend, torch, np)
    patch_pytorch_comparison_arraylike_contract(raw_pytorch, backend, torch)
    _patch_pytorch_dtype_promotion(raw_pytorch, backend, torch)


def _patch_pytorch_dtype_promotion(raw_pytorch, backend, torch) -> None:
    original_convert = raw_pytorch.convert_to_wider_dtype
    if getattr(original_convert, "_pyrecest_torch_promotion_contract", False):
        if getattr(backend, "__backend_name__", None) == "pytorch":
            backend.convert_to_wider_dtype = original_convert
        return

    def convert_to_wider_dtype(tensor_list):
        tensors = list(tensor_list)
        if not tensors:
            return tensors

        promoted_dtype = tensors[0].dtype
        for tensor in tensors[1:]:
            promoted_dtype = torch.promote_types(promoted_dtype, tensor.dtype)

        if all(tensor.dtype == promoted_dtype for tensor in tensors):
            return tensors
        return [raw_pytorch.cast(tensor, dtype=promoted_dtype) for tensor in tensors]

    convert_to_wider_dtype.__name__ = getattr(
        original_convert, "__name__", "convert_to_wider_dtype"
    )
    convert_to_wider_dtype.__doc__ = getattr(original_convert, "__doc__", None)
    convert_to_wider_dtype._pyrecest_torch_promotion_contract = True
    raw_pytorch.convert_to_wider_dtype = convert_to_wider_dtype
    if getattr(backend, "__backend_name__", None) == "pytorch":
        backend.convert_to_wider_dtype = convert_to_wider_dtype
