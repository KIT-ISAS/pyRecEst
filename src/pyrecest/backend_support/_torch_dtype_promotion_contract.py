"""PyTorch compatibility helpers for NumPy-style backend contracts."""

from __future__ import annotations


def _patch_pytorch_concatenate_axis_none_contract(raw_pytorch, torch_module, backend) -> None:
    """Make PyTorch concatenate flatten inputs when ``axis=None``."""
    original_concatenate = raw_pytorch.concatenate
    if getattr(original_concatenate, "_pyrecest_axis_none_contract", False):
        return

    def concatenate(seq, axis=0, out=None):
        tensors = [raw_pytorch.array(item) for item in seq]
        if tensors:
            tensors = raw_pytorch.convert_to_wider_dtype(tensors)
        if axis is None:
            tensors = [torch_module.flatten(tensor) for tensor in tensors]
            axis = 0
        result = torch_module.cat(tensors, dim=axis)
        if out is not None:
            out.copy_(result)
            return out
        return result

    concatenate.__name__ = getattr(original_concatenate, "__name__", "concatenate")
    concatenate.__doc__ = getattr(original_concatenate, "__doc__", None)
    concatenate._pyrecest_axis_none_contract = True
    raw_pytorch.concatenate = concatenate
    if getattr(backend, "__backend_name__", None) == "pytorch":
        backend.concatenate = concatenate


def patch_pytorch_dtype_promotion_contract() -> None:
    """Make PyTorch mixed-dtype helpers and concatenate match NumPy-style contracts."""
    try:
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
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

    _patch_pytorch_concatenate_axis_none_contract(raw_pytorch, torch, backend)
