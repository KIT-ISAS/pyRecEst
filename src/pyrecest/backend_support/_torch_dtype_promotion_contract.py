"""PyTorch backend compatibility helpers."""

from __future__ import annotations


def _patch_pytorch_allclose_equal_nan_contract(raw_pytorch, torch_module) -> None:
    """Make PyTorch allclose accept the NumPy/PyTorch equal_nan keyword."""

    original_allclose = raw_pytorch.allclose
    if getattr(original_allclose, "_pyrecest_equal_nan_contract", False):
        return

    default_atol = raw_pytorch.atol
    default_rtol = raw_pytorch.rtol

    def _as_allclose_tensor(value, *, device):
        if torch_module.is_tensor(value):
            if device is not None and value.device != device:
                return value.to(device=device)
            return value
        return torch_module.as_tensor(value, device=device)

    def allclose(a, b, atol=default_atol, rtol=default_rtol, equal_nan=False):
        device = next(
            (value.device for value in (a, b) if torch_module.is_tensor(value)),
            None,
        )
        a = _as_allclose_tensor(a, device=device)
        b = _as_allclose_tensor(b, device=device)
        a, b = raw_pytorch.convert_to_wider_dtype([a, b])
        a, b = torch_module.broadcast_tensors(a, b)
        return torch_module.allclose(
            a,
            b,
            atol=atol,
            rtol=rtol,
            equal_nan=equal_nan,
        )

    allclose.__name__ = getattr(original_allclose, "__name__", "allclose")
    allclose.__doc__ = getattr(original_allclose, "__doc__", None)
    allclose._pyrecest_equal_nan_contract = True
    raw_pytorch.allclose = allclose

    try:
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - import fails before this module
        return

    if getattr(backend, "__backend_name__", None) == "pytorch":
        backend.allclose = allclose


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

    _patch_pytorch_allclose_equal_nan_contract(raw_pytorch, torch)
