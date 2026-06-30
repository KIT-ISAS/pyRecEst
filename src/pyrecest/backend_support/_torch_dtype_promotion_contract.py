"""PyTorch backend compatibility helpers."""

from __future__ import annotations

from operator import index as _operator_index


def _pytorch_repeat_count(repetition) -> int:
    """Return one NumPy-style repeat count as a non-negative integer."""
    try:
        count = _operator_index(repetition)
    except TypeError as exc:
        raise TypeError("repeat counts must be integers") from exc
    if count < 0:
        raise ValueError("repeats may not contain negative values")
    return count


def _pytorch_repeat_counts(repeats, *, numpy_module, torch_module, device):
    """Normalize NumPy-style repeat counts for ``torch.repeat_interleave``."""
    if torch_module.is_tensor(repeats):
        if repeats.ndim > 1:
            raise ValueError("object too deep for desired array")
        if repeats.dtype.is_floating_point or repeats.dtype.is_complex:
            raise TypeError("repeat counts must be integers")
        repeat_counts = repeats.to(device=device, dtype=torch_module.long)
        if bool(torch_module.any(repeat_counts < 0)):
            raise ValueError("repeats may not contain negative values")
        return repeat_counts

    repeats_array = numpy_module.asarray(repeats)
    if repeats_array.shape == ():
        return _pytorch_repeat_count(repeats_array.item())
    if repeats_array.ndim > 1:
        raise ValueError("object too deep for desired array")
    if not numpy_module.can_cast(
        repeats_array.dtype,
        numpy_module.dtype("intp"),
        casting="safe",
    ):
        raise TypeError("repeat counts must be integers")
    repeat_counts = torch_module.as_tensor(
        repeats_array,
        dtype=torch_module.long,
        device=device,
    )
    if bool(torch_module.any(repeat_counts < 0)):
        raise ValueError("repeats may not contain negative values")
    return repeat_counts


def _patch_raw_pytorch_repeat_contract(raw_pytorch, torch_module) -> None:
    """Make raw PyTorch repeat follow PyRecEst's NumPy-style contract."""
    try:
        import numpy as numpy_module  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - NumPy is a core dependency
        return

    original_repeat = getattr(raw_pytorch, "repeat", None)
    if original_repeat is None or getattr(
        original_repeat,
        "_pyrecest_repeat_contract",
        False,
    ):
        return

    def repeat(a, repeats, axis=None, *, dim=None, output_size=None):
        if dim is not None:
            if axis is not None and axis != dim:
                raise TypeError("repeat() got both 'axis' and 'dim'")
            axis = dim
        if axis is not None:
            axis = _operator_index(axis)

        values = raw_pytorch.array(a)
        repeat_counts = _pytorch_repeat_counts(
            repeats,
            numpy_module=numpy_module,
            torch_module=torch_module,
            device=values.device,
        )
        kwargs = {"dim": axis}
        if output_size is not None:
            kwargs["output_size"] = output_size
        return original_repeat(values, repeat_counts, **kwargs)

    repeat.__name__ = getattr(original_repeat, "__name__", "repeat")
    repeat.__doc__ = getattr(original_repeat, "__doc__", None)
    repeat._pyrecest_repeat_contract = True
    raw_pytorch.repeat = repeat

    try:
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - import fails before backend exists
        return
    if getattr(backend, "__backend_name__", None) == "pytorch":
        backend.repeat = repeat


def patch_pytorch_dtype_promotion_contract() -> None:
    """Make PyTorch backend helpers follow PyRecEst compatibility contracts."""
    try:
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend import failed earlier
        return

    _patch_raw_pytorch_repeat_contract(raw_pytorch, torch)

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
