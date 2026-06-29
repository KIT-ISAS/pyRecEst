"""PyTorch backend compatibility helpers."""

from __future__ import annotations


def _patch_convert_to_wider_dtype(raw_pytorch, torch) -> None:
    """Make PyTorch mixed-dtype helpers use Torch's promotion rules."""

    original_convert = raw_pytorch.convert_to_wider_dtype
    if getattr(original_convert, "_pyrecest_torch_promotion_contract", False):
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


def _patch_arctan_array_like_contract(raw_pytorch, torch) -> None:
    """Make PyTorch arctan accept NumPy-style array-like inputs."""

    original_arctan = raw_pytorch.arctan
    if getattr(original_arctan, "_pyrecest_array_like_contract", False):
        return

    def arctan(x, *args, **kwargs):
        if not torch.is_tensor(x):
            x = raw_pytorch.array(x)
        return original_arctan(x, *args, **kwargs)

    arctan.__name__ = getattr(original_arctan, "__name__", "arctan")
    arctan.__doc__ = getattr(original_arctan, "__doc__", None)
    arctan._pyrecest_array_like_contract = True
    raw_pytorch.arctan = arctan

    try:
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - import fails before backend setup
        return
    if getattr(backend, "__backend_name__", None) == "pytorch":
        backend.arctan = arctan


def patch_pytorch_dtype_promotion_contract() -> None:
    """Patch PyTorch backend compatibility contracts."""

    try:
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend import failed earlier
        return

    _patch_convert_to_wider_dtype(raw_pytorch, torch)
    _patch_arctan_array_like_contract(raw_pytorch, torch)
