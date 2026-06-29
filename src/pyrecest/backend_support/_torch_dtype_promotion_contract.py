"""PyTorch compatibility helpers."""

from __future__ import annotations


def _pytorch_pad_width_pairs(pad_width, ndim, numpy_module):
    try:
        pad_pairs = numpy_module.broadcast_to(
            numpy_module.asarray(pad_width),
            (ndim, 2),
        )
    except ValueError as exc:
        raise ValueError(
            f"pad_width must be broadcastable to shape ({ndim}, 2)"
        ) from exc
    if numpy_module.any(pad_pairs < 0):
        raise ValueError("index can't contain negative values")
    return tuple((int(before), int(after)) for before, after in pad_pairs.tolist())


def _pytorch_edge_pad(values, pad_pairs, torch_module):
    result = values
    for axis, (pad_before, pad_after) in enumerate(pad_pairs):
        if pad_before == 0 and pad_after == 0:
            continue
        if result.shape[axis] == 0:
            raise ValueError(
                "can't extend empty axis "
                f"{axis} using modes other than 'constant' or 'empty'"
            )
        pieces = []
        if pad_before:
            first_indices = torch_module.zeros(
                pad_before,
                dtype=torch_module.long,
                device=result.device,
            )
            pieces.append(torch_module.index_select(result, axis, first_indices))
        pieces.append(result)
        if pad_after:
            last_index = result.shape[axis] - 1
            last_indices = torch_module.full(
                (pad_after,),
                last_index,
                dtype=torch_module.long,
                device=result.device,
            )
            pieces.append(torch_module.index_select(result, axis, last_indices))
        result = torch_module.cat(pieces, dim=axis)
    return result


def _patch_pytorch_pad_edge_contract(raw_pytorch, torch_module) -> None:
    """Make PyTorch ``pad`` accept NumPy's edge-padding mode."""
    original_pad = raw_pytorch.pad
    if getattr(original_pad, "_pyrecest_edge_pad_contract", False):
        return

    try:
        import numpy as np  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - import fails before this module
        return

    active_pytorch_backend = getattr(backend, "__backend_name__", None) == "pytorch"

    def pad(a, pad_width, mode="constant", constant_values=0.0):
        if mode != "edge":
            return original_pad(
                a,
                pad_width,
                mode=mode,
                constant_values=constant_values,
            )
        values = raw_pytorch.array(a)
        pad_pairs = _pytorch_pad_width_pairs(pad_width, values.ndim, np)
        return _pytorch_edge_pad(values, pad_pairs, torch_module)

    pad.__name__ = getattr(original_pad, "__name__", "pad")
    pad.__doc__ = getattr(original_pad, "__doc__", None)
    pad._pyrecest_edge_pad_contract = True
    raw_pytorch.pad = pad
    if active_pytorch_backend:
        backend.pad = pad


def patch_pytorch_dtype_promotion_contract() -> None:
    """Patch PyTorch backend compatibility gaps at backend-support import time."""
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
            original_convert,
            "__name__",
            "convert_to_wider_dtype",
        )
        convert_to_wider_dtype.__doc__ = getattr(original_convert, "__doc__", None)
        convert_to_wider_dtype._pyrecest_torch_promotion_contract = True
        raw_pytorch.convert_to_wider_dtype = convert_to_wider_dtype

    _patch_pytorch_pad_edge_contract(raw_pytorch, torch)
