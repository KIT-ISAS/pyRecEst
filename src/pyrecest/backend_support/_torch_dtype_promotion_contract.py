"""PyTorch backend compatibility helpers."""

from __future__ import annotations


def _sync_active_public_pytorch(name: str, helper) -> None:
    """Expose a patched raw helper through the active public PyTorch facade."""
    try:
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - import fails before this module
        return

    if getattr(backend, "__backend_name__", None) == "pytorch":
        setattr(backend, name, helper)


def _patch_pytorch_array_equal_numpy_contract(raw_pytorch, torch_module) -> None:
    """Make PyTorch array_equal accept NumPy's equal_nan keyword."""
    original_array_equal = raw_pytorch.array_equal
    if getattr(original_array_equal, "_pyrecest_numpy_contract", False):
        _sync_active_public_pytorch("array_equal", original_array_equal)
        return

    def array_equal(a, b, equal_nan=False):
        a = raw_pytorch.array(a)
        b = raw_pytorch.array(b)
        if tuple(a.shape) != tuple(b.shape):
            return False
        if not equal_nan:
            return torch_module.equal(a, b)
        equal_or_both_nan = torch_module.eq(a, b) | (
            torch_module.isnan(a) & torch_module.isnan(b)
        )
        return bool(torch_module.all(equal_or_both_nan))

    array_equal.__name__ = getattr(original_array_equal, "__name__", "array_equal")
    array_equal.__doc__ = getattr(original_array_equal, "__doc__", None)
    array_equal._pyrecest_numpy_contract = True
    raw_pytorch.array_equal = array_equal
    _sync_active_public_pytorch("array_equal", array_equal)


def patch_pytorch_dtype_promotion_contract() -> None:
    """Make PyTorch mixed-dtype helpers use Torch's promotion rules."""
    try:
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend import failed earlier
        return

    _patch_pytorch_array_equal_numpy_contract(raw_pytorch, torch)

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
