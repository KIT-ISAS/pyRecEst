"""Shared NumPy ``assignment_by_sum`` duplicate-index compatibility hook."""

from __future__ import annotations

import sys


def _install_assignment_by_sum(assignment_by_sum, backend, shared_numpy) -> None:
    """Install the patched helper on loaded shared-NumPy facade modules."""

    shared_numpy.assignment_by_sum = assignment_by_sum
    for module_name in ("pyrecest._backend.numpy", "pyrecest._backend.autograd"):
        module = sys.modules.get(module_name)
        if module is not None:
            module.assignment_by_sum = assignment_by_sum
    if getattr(backend, "__backend_name__", None) in {"numpy", "autograd"}:
        backend.assignment_by_sum = assignment_by_sum


def patch_shared_numpy_assignment_by_sum_duplicate_indices() -> None:
    """Make shared NumPy assignment-by-sum accumulate repeated indices."""

    try:
        import pyrecest._backend._shared_numpy as shared_numpy  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - backend import failure path
        return

    if getattr(backend, "__backend_name__", None) not in {"numpy", "autograd"}:
        return

    original_assignment_by_sum = shared_numpy.assignment_by_sum
    if getattr(
        original_assignment_by_sum,
        "_pyrecest_duplicate_index_accumulation_contract",
        False,
    ):
        _install_assignment_by_sum(original_assignment_by_sum, backend, shared_numpy)
        return

    def assignment_by_sum(x, values, indices, axis=0):
        x_new = shared_numpy.copy(shared_numpy.array(x))

        if shared_numpy._is_empty_index_sequence(indices):
            return x_new

        use_vectorization = hasattr(indices, "__len__") and len(indices) < shared_numpy.ndim(
            x_new
        )
        if shared_numpy._is_boolean(indices):
            x_new[indices] += values
            return x_new

        zip_indices = shared_numpy._is_iterable(indices) and shared_numpy._is_iterable(
            indices[0]
        )
        if zip_indices:
            indices = tuple(zip(*indices))
        if not use_vectorization:
            len_indices = len(indices) if shared_numpy._is_iterable(indices) else 1
            len_values = len(values) if shared_numpy._is_iterable(values) else 1
            if len_values > 1 and len_values != len_indices:
                raise ValueError("Either one value or as many values as indices")
            shared_numpy._np.add.at(x_new, indices, values)
        else:
            indices = tuple(list(indices[:axis]) + [slice(None)] + list(indices[axis:]))
            x_new[indices] += values
        return x_new

    assignment_by_sum.__name__ = getattr(
        original_assignment_by_sum,
        "__name__",
        "assignment_by_sum",
    )
    assignment_by_sum.__doc__ = getattr(original_assignment_by_sum, "__doc__", None)
    assignment_by_sum._pyrecest_duplicate_index_accumulation_contract = True

    _install_assignment_by_sum(assignment_by_sum, backend, shared_numpy)


__all__ = ["patch_shared_numpy_assignment_by_sum_duplicate_indices"]
