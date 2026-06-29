"""PyTorch backend compatibility helpers."""

from __future__ import annotations


def _patch_stack_helpers(raw_pytorch, torch_module) -> None:
    """Make low-level stack helpers accept NumPy-style array-like inputs."""
    try:
        import numpy as np  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - NumPy is a required dependency
        return

    def _tensor_sequence(tup):
        return [raw_pytorch.array(item) for item in tup]

    def hstack(tup):
        tensors = [torch_module.atleast_1d(tensor) for tensor in _tensor_sequence(tup)]
        if not tensors:
            return torch_module.cat(tensors, dim=0)
        return torch_module.cat(tensors, dim=0 if tensors[0].ndim == 1 else 1)

    def vstack(tup):
        tensors = [torch_module.atleast_2d(tensor) for tensor in _tensor_sequence(tup)]
        return torch_module.cat(tensors, dim=0)

    def column_stack(tup):
        tensors = []
        for tensor in _tensor_sequence(tup):
            if tensor.ndim < 2:
                tensor = tensor.reshape(-1, 1)
            tensors.append(tensor)
        return torch_module.cat(tensors, dim=1)

    def dstack(tup):
        tensors = [torch_module.atleast_3d(tensor) for tensor in _tensor_sequence(tup)]
        return torch_module.cat(tensors, dim=2)

    for helper_name, helper in {
        "hstack": hstack,
        "vstack": vstack,
        "column_stack": column_stack,
        "dstack": dstack,
    }.items():
        current_helper = getattr(raw_pytorch, helper_name)
        if getattr(current_helper, "_pyrecest_numpy_contract", False):
            continue
        helper.__name__ = helper_name
        helper.__doc__ = getattr(np, helper_name).__doc__
        helper._pyrecest_numpy_contract = True
        setattr(raw_pytorch, helper_name, helper)


def patch_pytorch_dtype_promotion_contract() -> None:
    """Make PyTorch mixed-dtype helpers use Torch's promotion rules."""
    try:
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend import failed earlier
        return

    original_convert = raw_pytorch.convert_to_wider_dtype
    if not getattr(original_convert, "_pyrecest_torch_promotion_contract", False):

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

    _patch_stack_helpers(raw_pytorch, torch)
