"""PyTorch dtype promotion compatibility helpers."""

from __future__ import annotations

from operator import index as _operator_index


def patch_pytorch_dtype_promotion_contract() -> None:
    """Make PyTorch mixed-dtype helpers use Torch's promotion rules."""
    try:
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend import failed earlier
        return

    _patch_pytorch_diff_numpy_contract(raw_pytorch, torch)

    original_convert = raw_pytorch.convert_to_wider_dtype
    if getattr(original_convert, "_pyrecest_torch_promotion_contract", False):
        if getattr(backend, "__backend_name__", None) == "pytorch":
            backend.convert_to_wider_dtype = original_convert
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
    if getattr(backend, "__backend_name__", None) == "pytorch":
        backend.convert_to_wider_dtype = convert_to_wider_dtype


def _patch_pytorch_diff_numpy_contract(raw_pytorch, torch) -> None:
    """Make raw/public PyTorch diff follow the PyRecEst NumPy-style contract."""
    try:
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - import fails before this module
        backend = None

    original_diff = raw_pytorch.diff
    if getattr(original_diff, "_pyrecest_numpy_contract", False):
        return
    no_boundary = object()

    def _normalize_axis(axis, ndim):
        axis = _operator_index(axis)
        if axis < 0:
            axis += ndim
        if axis < 0 or axis >= ndim:
            raise IndexError(f"axis {axis} is out of bounds for array of dimension {ndim}")
        return axis

    def _boundary(value, reference, axis):
        boundary = raw_pytorch.array(value)
        if boundary.device != reference.device:
            boundary = boundary.to(device=reference.device)
        if boundary.ndim == 0:
            boundary_shape = list(reference.shape)
            boundary_shape[axis] = 1
            boundary = torch.broadcast_to(boundary, tuple(boundary_shape))
        return boundary

    def diff(a, n=1, axis=-1, prepend=no_boundary, append=no_boundary):
        values = raw_pytorch.array(a)
        order = _operator_index(n)
        if order < 0:
            raise ValueError(f"order must be non-negative but got {order}")
        if order == 0:
            return values.clone()
        if values.ndim == 0:
            raise ValueError("diff requires input that is at least one dimensional")

        axis = _normalize_axis(axis, values.ndim)
        diff_inputs = []
        if prepend is not no_boundary:
            diff_inputs.append(_boundary(prepend, values, axis))
        diff_inputs.append(values)
        if append is not no_boundary:
            diff_inputs.append(_boundary(append, values, axis))
        if len(diff_inputs) > 1:
            diff_inputs = raw_pytorch.convert_to_wider_dtype(diff_inputs)
            values = torch.cat(diff_inputs, dim=axis)
        return torch.diff(values, n=order, dim=axis)

    diff.__name__ = getattr(original_diff, "__name__", "diff")
    diff.__doc__ = getattr(original_diff, "__doc__", None)
    diff._pyrecest_numpy_contract = True
    raw_pytorch.diff = diff
    if backend is not None and getattr(backend, "__backend_name__", None) == "pytorch":
        backend.diff = diff
