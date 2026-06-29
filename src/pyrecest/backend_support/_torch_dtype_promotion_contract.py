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


def _like_input(raw_pytorch, torch, value):
    return value if torch.is_tensor(value) else raw_pytorch.array(value)


def _normalize_like_dtype(raw_pytorch, dtype):
    if dtype is None:
        return None
    return raw_pytorch.array((), dtype=dtype).dtype


def _like_kwargs(raw_pytorch, dtype, kwargs):
    kwargs = dict(kwargs)
    if dtype is not None:
        kwargs["dtype"] = _normalize_like_dtype(raw_pytorch, dtype)
    return kwargs


def _wrap_like_helper(raw_pytorch, torch, original, helper_name):
    if getattr(original, "_pyrecest_numpy_contract", False):
        return original

    def helper(a, dtype=None, **kwargs):
        return original(
            _like_input(raw_pytorch, torch, a),
            **_like_kwargs(raw_pytorch, dtype, kwargs),
        )

    helper.__name__ = getattr(original, "__name__", helper_name)
    helper.__doc__ = getattr(original, "__doc__", None)
    helper._pyrecest_numpy_contract = True
    return helper


def _wrap_full_like_helper(raw_pytorch, torch, original):
    if getattr(original, "_pyrecest_numpy_contract", False):
        return original

    def full_like(a, fill_value, dtype=None, **kwargs):
        return original(
            _like_input(raw_pytorch, torch, a),
            fill_value,
            **_like_kwargs(raw_pytorch, dtype, kwargs),
        )

    full_like.__name__ = getattr(original, "__name__", "full_like")
    full_like.__doc__ = getattr(original, "__doc__", None)
    full_like._pyrecest_numpy_contract = True
    return full_like


def _patch_like_helpers(raw_pytorch, torch) -> None:
    """Make PyTorch ``*_like`` helpers accept NumPy-style array-like inputs."""
    try:
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - import fails before this module
        backend = None

    active_pytorch_backend = (
        backend is not None and getattr(backend, "__backend_name__", None) == "pytorch"
    )

    for helper_name in ("empty_like", "ones_like", "zeros_like"):
        patched = _wrap_like_helper(
            raw_pytorch,
            torch,
            getattr(raw_pytorch, helper_name),
            helper_name,
        )
        setattr(raw_pytorch, helper_name, patched)
        if active_pytorch_backend:
            setattr(backend, helper_name, patched)

    patched_full_like = _wrap_full_like_helper(raw_pytorch, torch, raw_pytorch.full_like)
    raw_pytorch.full_like = patched_full_like
    if active_pytorch_backend:
        backend.full_like = patched_full_like


def patch_pytorch_dtype_promotion_contract() -> None:
    """Patch low-level PyTorch backend helpers to honor PyRecEst contracts."""
    try:
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend import failed earlier
        return

    _patch_convert_to_wider_dtype(raw_pytorch, torch)
    _patch_like_helpers(raw_pytorch, torch)
