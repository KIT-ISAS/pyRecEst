"""PyTorch sort compatibility helpers."""

from __future__ import annotations

from operator import index as _operator_index

_SORT_KINDS = {"quicksort", "heapsort", "mergesort", "stable"}
_STABLE_SORT_KINDS = {"mergesort", "stable"}


def patch_pytorch_sort_numpy_contract() -> None:
    """Make PyTorch sort follow NumPy axis and stable-kind contracts."""
    try:
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend may be unavailable
        return

    original_sort = raw_pytorch.sort
    if getattr(original_sort, "_pyrecest_numpy_sort_contract", False):
        return

    def sort(a, axis=-1, kind=None, order=None, *, stable=None, descending=False):
        if order is not None:
            raise TypeError("PyTorch backend sort does not support field-order sorting")
        if kind is not None:
            if kind not in _SORT_KINDS:
                raise ValueError(
                    "sort kind must be one of 'quicksort', 'heapsort', 'mergesort', or 'stable'"
                )
            if kind in _STABLE_SORT_KINDS:
                if stable is False:
                    raise TypeError("sort() got inconsistent 'kind' and 'stable' arguments")
                stable = True
        stable = bool(stable) if stable is not None else False

        values = raw_pytorch.array(a)
        if axis is None:
            values = torch.flatten(values)
            axis = -1
        else:
            axis = _operator_index(axis)
        sorted_values, _ = torch.sort(
            values,
            dim=axis,
            stable=stable,
            descending=descending,
        )
        return sorted_values

    sort.__name__ = getattr(original_sort, "__name__", "sort")
    sort.__doc__ = getattr(original_sort, "__doc__", None)
    sort._pyrecest_numpy_sort_contract = True
    raw_pytorch.sort = sort
    if getattr(backend, "__backend_name__", None) == "pytorch":
        backend.sort = sort
