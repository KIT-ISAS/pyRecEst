"""PyTorch backend compatibility helpers."""

from __future__ import annotations

from operator import index as _operator_index


def patch_pytorch_dtype_promotion_contract() -> None:
    """Make PyTorch mixed-dtype helpers use Torch's promotion rules."""
    try:
        import numpy as np  # pylint: disable=import-outside-toplevel
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend import failed earlier
        return

    _patch_pytorch_assignment_numpy_boolean_contract(raw_pytorch, torch, np)
    _patch_pytorch_diff_numpy_contract(raw_pytorch, torch)
    _patch_pytorch_pad_constant_values_contract(raw_pytorch, torch, np)

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


def _normalize_numpy_boolean_index(indices, torch, np):
    """Convert NumPy boolean indices before PyTorch assignment normalization."""
    if isinstance(indices, np.bool_):
        return bool(indices)
    if isinstance(indices, np.ndarray) and indices.dtype == np.bool_:
        return torch.as_tensor(indices, dtype=torch.bool)
    return indices


def _patch_pytorch_assignment_numpy_boolean_contract(raw_pytorch, torch, np) -> None:
    """Make PyTorch assignment helpers treat NumPy boolean arrays as masks."""
    try:
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - import fails before this module
        backend = None

    def _wrap_assignment(original_assignment):
        if getattr(original_assignment, "_pyrecest_numpy_boolean_index_contract", False):
            return original_assignment

        def assignment(x, values, indices, axis=0):
            indices = _normalize_numpy_boolean_index(indices, torch, np)
            return original_assignment(x, values, indices, axis=axis)

        assignment.__name__ = getattr(original_assignment, "__name__", "assignment")
        assignment.__doc__ = getattr(original_assignment, "__doc__", None)
        assignment._pyrecest_numpy_boolean_index_contract = True
        return assignment

    raw_pytorch.assignment = _wrap_assignment(raw_pytorch.assignment)
    raw_pytorch.assignment_by_sum = _wrap_assignment(raw_pytorch.assignment_by_sum)
    if backend is not None and getattr(backend, "__backend_name__", None) == "pytorch":
        backend.assignment = raw_pytorch.assignment
        backend.assignment_by_sum = raw_pytorch.assignment_by_sum


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


def _normalize_pad_pairs(pad_width, ndim, np):
    """Return NumPy-style per-axis pad-width pairs as Python integers."""
    try:
        pad_pairs = np.broadcast_to(np.asarray(pad_width), (ndim, 2))
    except ValueError as exc:
        raise ValueError(f"pad_width must be broadcastable to shape ({ndim}, 2)") from exc
    if np.any(pad_pairs < 0):
        raise ValueError("index can't contain negative values")
    try:
        return tuple(
            (_operator_index(before), _operator_index(after))
            for before, after in pad_pairs.tolist()
        )
    except TypeError as exc:
        raise TypeError("pad_width must be of integral type") from exc


def _normalize_constant_value_pairs(constant_values, ndim, np):
    """Return NumPy-style per-axis constant-value pairs."""
    try:
        constant_pairs = np.broadcast_to(np.asarray(constant_values), (ndim, 2))
    except ValueError as exc:
        raise ValueError(
            f"constant_values must be broadcastable to shape ({ndim}, 2)"
        ) from exc
    return tuple(tuple(pair) for pair in constant_pairs.tolist())


def _filled_pad_block(shape, value, reference, torch):
    """Return a constant-filled block compatible with ``reference``."""
    scalar_value = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    if scalar_value.ndim != 0:
        raise ValueError("constant_values entries must be scalar")
    block = torch.empty(tuple(shape), dtype=reference.dtype, device=reference.device)
    block.fill_(scalar_value)
    return block


def _constant_pad(values, pad_width, constant_values, torch, np):
    """Pad a tensor with NumPy-style per-axis constant values."""
    pad_pairs = _normalize_pad_pairs(pad_width, values.ndim, np)
    constant_pairs = _normalize_constant_value_pairs(constant_values, values.ndim, np)
    result = values
    for axis, ((before, after), (before_value, after_value)) in enumerate(
        zip(pad_pairs, constant_pairs)
    ):
        if before:
            before_shape = list(result.shape)
            before_shape[axis] = before
            before_block = _filled_pad_block(before_shape, before_value, result, torch)
            result = torch.cat((before_block, result), dim=axis)
        if after:
            after_shape = list(result.shape)
            after_shape[axis] = after
            after_block = _filled_pad_block(after_shape, after_value, result, torch)
            result = torch.cat((result, after_block), dim=axis)
    return result


def _patch_pytorch_pad_constant_values_contract(raw_pytorch, torch, np) -> None:
    """Make raw/public PyTorch pad accept NumPy-style constant_values."""
    try:
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - import fails before this module
        backend = None

    original_pad = raw_pytorch.pad
    if getattr(original_pad, "_pyrecest_constant_values_contract", False):
        return

    def pad(a, pad_width, mode="constant", constant_values=0.0):
        values = raw_pytorch.array(a)
        if mode != "constant":
            return original_pad(
                values,
                pad_width,
                mode=mode,
                constant_values=constant_values,
            )
        return _constant_pad(values, pad_width, constant_values, torch, np)

    pad.__name__ = getattr(original_pad, "__name__", "pad")
    pad.__doc__ = getattr(original_pad, "__doc__", None)
    pad._pyrecest_constant_values_contract = True
    raw_pytorch.pad = pad
    if backend is not None and getattr(backend, "__backend_name__", None) == "pytorch":
        backend.pad = pad
