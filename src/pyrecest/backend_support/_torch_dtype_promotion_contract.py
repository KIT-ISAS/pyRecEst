"""PyTorch dtype promotion compatibility helpers."""

from __future__ import annotations

from operator import index as _operator_index


def _normalize_pytorch_squeeze_axes(axis, ndim: int) -> tuple[int, ...]:
    """Normalize NumPy-style squeeze axes for the PyTorch backend."""

    try:
        axes = (_operator_index(axis),)
    except TypeError:
        if isinstance(axis, (str, bytes)):
            raise TypeError(
                "squeeze axis must be an integer or a sequence of integers"
            ) from None
        try:
            axes = tuple(_operator_index(one_axis) for one_axis in axis)
        except TypeError as exc:
            raise TypeError(
                "squeeze axis must be an integer or a sequence of integers"
            ) from exc

    normalized_axes = []
    for one_axis in axes:
        normalized_axis = one_axis + ndim if one_axis < 0 else one_axis
        if normalized_axis < 0 or normalized_axis >= ndim:
            raise IndexError(
                f"axis {one_axis} is out of bounds for array of dimension {ndim}"
            )
        normalized_axes.append(normalized_axis)

    if len(set(normalized_axes)) != len(normalized_axes):
        raise ValueError("duplicate value in 'axis'")
    return tuple(normalized_axes)


def _patch_pytorch_squeeze_axis_contract() -> None:
    """Make PyTorch squeeze accept NumPy-style tuple axes."""
    try:
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend import failed earlier
        return

    original_squeeze = raw_pytorch.squeeze
    if getattr(original_squeeze, "_pyrecest_squeeze_axis_contract", False):
        return

    def squeeze(x, axis=None):
        x = raw_pytorch.array(x)
        if axis is None:
            return torch.squeeze(x)

        axes = _normalize_pytorch_squeeze_axes(axis, x.ndim)
        if not axes:
            return x
        if any(x.shape[one_axis] != 1 for one_axis in axes):
            return x

        result = x
        for one_axis in sorted(axes, reverse=True):
            result = torch.squeeze(result, dim=one_axis)
        return result

    squeeze.__name__ = getattr(original_squeeze, "__name__", "squeeze")
    squeeze.__doc__ = getattr(original_squeeze, "__doc__", None)
    squeeze._pyrecest_squeeze_axis_contract = True
    raw_pytorch.squeeze = squeeze

    try:
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - import fails before this module
        return
    if getattr(backend, "__backend_name__", None) == "pytorch":
        backend.squeeze = squeeze


def patch_pytorch_dtype_promotion_contract() -> None:
    """Make PyTorch mixed-dtype helpers use Torch's promotion rules."""
    _patch_pytorch_squeeze_axis_contract()

    try:
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend import failed earlier
        return

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
