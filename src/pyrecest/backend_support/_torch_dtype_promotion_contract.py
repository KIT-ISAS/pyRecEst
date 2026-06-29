"""PyTorch backend compatibility helpers."""

from __future__ import annotations

from operator import index as _operator_index


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


def _as_int_tuple(value, np, torch, name):
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    value_array = np.asarray(value)
    if value_array.ndim == 0:
        try:
            return (_operator_index(value_array.item()),)
        except TypeError as exc:
            raise TypeError(f"{name} must contain integers") from exc
    if value_array.ndim != 1:
        raise ValueError(f"{name} must be a scalar or 1-D sequence")
    try:
        return tuple(_operator_index(one_value) for one_value in value_array.tolist())
    except TypeError as exc:
        raise TypeError(f"{name} must contain integers") from exc


def _normalized_axes(axis, ndim, np, torch):
    axes = _as_int_tuple(axis, np, torch, "axis")
    normalized_axes = tuple(
        one_axis + ndim if one_axis < 0 else one_axis for one_axis in axes
    )
    for original_axis, normalized_axis in zip(axes, normalized_axes):
        if normalized_axis < 0 or normalized_axis >= ndim:
            raise IndexError(
                f"axis {original_axis} is out of bounds for array of dimension {ndim}"
            )
    return normalized_axes


def _roll_pairs(shift, axis, ndim, np, torch):
    shifts = _as_int_tuple(shift, np, torch, "shift")
    axes = _normalized_axes(axis, ndim, np, torch)
    if not shifts or not axes:
        return (), ()
    try:
        broadcast = np.broadcast(shifts, axes)
    except ValueError as exc:
        raise ValueError("shift and axis are not broadcast-compatible") from exc

    shift_by_axis = {}
    for one_shift, one_axis in broadcast:
        shift_by_axis[one_axis] = shift_by_axis.get(one_axis, 0) + int(one_shift)
    return tuple(shift_by_axis.values()), tuple(shift_by_axis.keys())


def _patch_roll_numpy_contract(raw_pytorch, torch) -> None:
    """Make PyTorch roll accept NumPy-style array-like inputs."""

    import numpy as np  # pylint: disable=import-outside-toplevel

    original_roll = raw_pytorch.roll
    if getattr(original_roll, "_pyrecest_numpy_contract", False):
        return

    def roll(a, shift=None, axis=None, *, shifts=None, dims=None):
        if shifts is not None:
            if shift is not None:
                raise TypeError("roll() got both 'shift' and 'shifts'")
            shift = shifts
        if shift is None:
            raise TypeError("roll() missing required argument 'shift'")
        if dims is not None:
            if axis is not None and axis != dims:
                raise TypeError("roll() got both 'axis' and 'dims'")
            axis = dims

        values = raw_pytorch.array(a)
        if axis is None:
            shift_values = _as_int_tuple(shift, np, torch, "shift")
            if not shift_values:
                return values.clone()
            flattened = values.reshape(-1)
            return torch.roll(flattened, sum(shift_values), 0).reshape(
                tuple(values.shape)
            )

        roll_shifts, roll_axes = _roll_pairs(shift, axis, values.ndim, np, torch)
        if not roll_shifts:
            return values.clone()
        return torch.roll(values, roll_shifts, roll_axes)

    roll.__name__ = getattr(original_roll, "__name__", "roll")
    roll.__doc__ = getattr(np.roll, "__doc__", None)
    roll._pyrecest_numpy_contract = True
    raw_pytorch.roll = roll

    try:
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - import fails before backend setup
        return
    if getattr(backend, "__backend_name__", None) == "pytorch":
        backend.roll = roll


def patch_pytorch_dtype_promotion_contract() -> None:
    """Patch low-level PyTorch backend helpers to honor PyRecEst contracts."""
    try:
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend import failed earlier
        return

    _patch_convert_to_wider_dtype(raw_pytorch, torch)
    _patch_roll_numpy_contract(raw_pytorch, torch)
