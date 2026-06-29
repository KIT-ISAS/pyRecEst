"""PyTorch backend compatibility helpers."""

from __future__ import annotations

from operator import index as _operator_index


def _patch_convert_to_wider_dtype_contract(raw_pytorch, torch) -> None:
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


def _resolve_argsort_axis(axis, dim):
    """Resolve NumPy's ``axis`` and PyTorch's ``dim`` aliases."""

    if dim is not None:
        if axis != -1 and axis != dim:
            raise TypeError("argsort() got both 'axis' and 'dim'")
        axis = dim
    if axis is None:
        return None
    return _operator_index(axis)


def _resolve_argsort_stability(kind, stable) -> bool:
    """Resolve NumPy-style sorting stability arguments for ``torch.argsort``."""

    if kind is not None:
        valid_kinds = {"quicksort", "heapsort", "mergesort", "stable"}
        if kind not in valid_kinds:
            raise ValueError(f"sort kind must be one of {sorted(valid_kinds)!r}")
        if kind in {"mergesort", "stable"} and stable is None:
            stable = True
    return bool(stable) if stable is not None else False


def _patch_argsort_numpy_contract(raw_pytorch, torch, backend) -> None:
    """Make PyTorch argsort accept NumPy-style array-like inputs and axis."""

    original_argsort = raw_pytorch.argsort
    if getattr(original_argsort, "_pyrecest_numpy_contract", False):
        return

    def argsort(
        a,
        axis=-1,
        kind=None,
        order=None,
        stable=None,
        *,
        dim=None,
        descending=False,
    ):
        if order is not None:
            raise TypeError("argsort() got an unsupported 'order' argument")

        values = raw_pytorch.array(a)
        resolved_axis = _resolve_argsort_axis(axis, dim)
        if resolved_axis is None:
            values = values.reshape(-1)
            resolved_axis = -1
        return torch.argsort(
            values,
            dim=resolved_axis,
            descending=descending,
            stable=_resolve_argsort_stability(kind, stable),
        )

    argsort.__name__ = getattr(original_argsort, "__name__", "argsort")
    argsort.__doc__ = getattr(original_argsort, "__doc__", None)
    argsort._pyrecest_numpy_contract = True
    raw_pytorch.argsort = argsort
    if getattr(backend, "__backend_name__", None) == "pytorch":
        backend.argsort = argsort


def patch_pytorch_dtype_promotion_contract() -> None:
    """Apply PyTorch backend compatibility patches used during package import."""

    try:
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend import failed earlier
        return

    _patch_convert_to_wider_dtype_contract(raw_pytorch, torch)
    _patch_argsort_numpy_contract(raw_pytorch, torch, backend)
