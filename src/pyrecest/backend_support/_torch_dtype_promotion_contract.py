"""PyTorch backend compatibility helpers."""

from __future__ import annotations


LIKE_CREATION_HELPERS = ("empty_like", "full_like", "ones_like", "zeros_like")


def _patch_pytorch_convert_to_wider_dtype(raw_pytorch, torch) -> None:
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


def _wrap_pytorch_like_creation(original_like, raw_pytorch, helper_name):
    """Return a ``*_like`` helper accepting NumPy-style array-like inputs."""
    if getattr(original_like, "_pyrecest_like_arraylike_contract", False):
        return original_like

    def like_creation(a, *args, **kwargs):
        return original_like(raw_pytorch.array(a), *args, **kwargs)

    like_creation.__name__ = getattr(original_like, "__name__", helper_name)
    like_creation.__doc__ = getattr(original_like, "__doc__", None)
    like_creation._pyrecest_like_arraylike_contract = True
    return like_creation


def _patch_pytorch_like_creation_contract(raw_pytorch) -> None:
    """Make raw PyTorch ``*_like`` helpers accept array-like inputs."""
    try:
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - import failed before this helper
        backend = None

    active_pytorch_backend = getattr(backend, "__backend_name__", None) == "pytorch"

    for helper_name in LIKE_CREATION_HELPERS:
        original_like = getattr(raw_pytorch, helper_name, None)
        if original_like is None:
            continue
        wrapped_like = _wrap_pytorch_like_creation(
            original_like,
            raw_pytorch,
            helper_name,
        )
        setattr(raw_pytorch, helper_name, wrapped_like)
        if active_pytorch_backend:
            setattr(backend, helper_name, wrapped_like)


def patch_pytorch_dtype_promotion_contract() -> None:
    """Patch PyTorch backend helpers that need PyRecEst compatibility shims."""
    try:
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend import failed earlier
        return

    _patch_pytorch_convert_to_wider_dtype(raw_pytorch, torch)
    _patch_pytorch_like_creation_contract(raw_pytorch)
