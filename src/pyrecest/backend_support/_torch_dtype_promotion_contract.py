"""PyTorch dtype promotion compatibility helpers."""

from __future__ import annotations


def _patch_pytorch_isclose_equal_nan_contract(raw_pytorch, torch_module) -> None:
    """Make PyTorch isclose accept NumPy's equal_nan keyword."""

    original_isclose = raw_pytorch.isclose
    if getattr(original_isclose, "_pyrecest_equal_nan_contract", False):
        return

    default_atol = raw_pytorch.atol
    default_rtol = raw_pytorch.rtol

    def _as_isclose_tensor(value, *, device):
        tensor = value if torch_module.is_tensor(value) else raw_pytorch.array(value)
        if device is not None and tensor.device != device:
            return tensor.to(device=device)
        return tensor

    def isclose(x, y, rtol=default_rtol, atol=default_atol, equal_nan=False):
        device = next(
            (value.device for value in (x, y) if torch_module.is_tensor(value)),
            None,
        )
        x = _as_isclose_tensor(x, device=device)
        y = _as_isclose_tensor(y, device=device)
        x, y = raw_pytorch.convert_to_wider_dtype([x, y])
        return torch_module.isclose(
            x,
            y,
            atol=atol,
            rtol=rtol,
            equal_nan=equal_nan,
        )

    isclose.__name__ = getattr(original_isclose, "__name__", "isclose")
    isclose.__doc__ = getattr(original_isclose, "__doc__", None)
    isclose._pyrecest_equal_nan_contract = True
    raw_pytorch.isclose = isclose

    try:
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - import fails before this module
        return

    if getattr(backend, "__backend_name__", None) == "pytorch":
        backend.isclose = isclose


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

    _patch_pytorch_isclose_equal_nan_contract(raw_pytorch, torch)
