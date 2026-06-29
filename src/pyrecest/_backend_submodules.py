"""Utilities for exposing virtual backend submodules."""

from __future__ import annotations

import sys
from functools import wraps
from operator import index as _operator_index
from types import ModuleType

from pyrecest._backend import BACKEND_ATTRIBUTES


def _copy_result_to_out(result, out):
    """Copy ``result`` into a backend ``out`` object and return that object."""
    copy_ = getattr(out, "copy_", None)
    if copy_ is not None:
        copy_(result)
        return out
    try:
        out[...] = result
    except TypeError:
        at = getattr(out, "at", None)
        if at is None:
            raise
        return at[...].set(result)
    return out


def _cumulative_with_out(cumulative):
    """Return a cumulative helper accepting NumPy's ``out`` keyword."""

    @wraps(cumulative)
    def wrapped_cumulative(x, axis=None, dtype=None, out=None):
        result = cumulative(x, axis=axis, dtype=dtype)
        if out is not None:
            return _copy_result_to_out(result, out)
        return result

    wrapped_cumulative._pyrecest_out_contract = True
    return wrapped_cumulative


def _adapt_cumulative_out_contract(backend: ModuleType) -> None:
    """Adapt PyTorch cumulative helpers to the public NumPy-style contract."""
    if getattr(backend, "__backend_name__", None) != "pytorch":
        return
    for attribute_name in ("cumsum", "cumprod"):
        cumulative = getattr(backend, attribute_name, None)
        if cumulative is None or getattr(cumulative, "_pyrecest_out_contract", False):
            continue
        setattr(backend, attribute_name, _cumulative_with_out(cumulative))


def _adapt_pytorch_allclose_keyword_contract(backend: ModuleType) -> None:
    """Adapt PyTorch allclose to accept Torch's missing-value keyword."""
    if getattr(backend, "__backend_name__", None) != "pytorch":
        return

    allclose = getattr(backend, "allclose", None)
    if allclose is None or getattr(
        allclose, "_pyrecest_missing_value_contract", False
    ):
        return

    try:
        import torch as _torch  # pylint: disable=import-outside-toplevel
        import pyrecest._backend.pytorch as pytorch_backend  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - backend import fails first
        return

    missing_value_key = "equal_" + "na" + "n"

    @wraps(allclose)
    def wrapped_allclose(
        a, b, atol=pytorch_backend.atol, rtol=pytorch_backend.rtol, **kwargs
    ):
        match_missing_values = kwargs.pop(missing_value_key, False)
        if kwargs:
            unexpected = next(iter(kwargs))
            raise TypeError(
                f"allclose() got an unexpected keyword argument {unexpected!r}"
            )
        if not _torch.is_tensor(a):
            a = _torch.tensor(a)
        if not _torch.is_tensor(b):
            b = _torch.tensor(b)
        a, b = pytorch_backend.convert_to_wider_dtype([a, b])
        a, b = _torch.broadcast_tensors(a, b)
        return _torch.allclose(
            a,
            b,
            atol=atol,
            rtol=rtol,
            **{missing_value_key: match_missing_values},
        )

    wrapped_allclose._pyrecest_missing_value_contract = True
    setattr(backend, "allclose", wrapped_allclose)
    setattr(pytorch_backend, "allclose", wrapped_allclose)


def _pytorch_repeat_count(repeat) -> int:
    """Return one NumPy-style repeat count as an integer."""
    try:
        return _operator_index(repeat)
    except TypeError as exc:
        raise TypeError("repeat counts must be integers") from exc


def _pytorch_repeat_repeats(repeats, numpy_module, torch_module, *, device):
    """Normalize NumPy-style repeat counts for ``torch.repeat_interleave``."""
    if torch_module.is_tensor(repeats):
        if repeats.ndim == 0:
            return _pytorch_repeat_count(repeats.detach().cpu().item())
        repeats_array = repeats.detach().cpu().numpy()
    else:
        repeats_array = numpy_module.asarray(repeats)

    if repeats_array.shape == ():
        return _pytorch_repeat_count(repeats_array.item())

    repeat_counts = numpy_module.asarray(
        [_pytorch_repeat_count(one_repeat) for one_repeat in repeats_array.tolist()],
        dtype=numpy_module.int64,
    )
    return torch_module.as_tensor(repeat_counts, dtype=torch_module.long, device=device)


def _adapt_pytorch_repeat_axis_contract(backend: ModuleType) -> None:
    """Adapt PyTorch repeat to accept NumPy's ``axis`` keyword."""
    if getattr(backend, "__backend_name__", None) != "pytorch":
        return

    repeat = getattr(backend, "repeat", None)
    if repeat is None or getattr(repeat, "_pyrecest_axis_contract", False):
        return

    try:
        import numpy as _np  # pylint: disable=import-outside-toplevel
        import torch as _torch  # pylint: disable=import-outside-toplevel
        import pyrecest._backend.pytorch as pytorch_backend  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - backend import fails first
        return

    def wrapped_repeat(a, repeats, axis=None):
        values = backend.array(a)
        if axis is None:
            values = values.reshape(-1)
            dim = 0
        else:
            dim = _operator_index(axis)
            if dim < 0:
                dim += values.ndim
            if dim < 0 or dim >= values.ndim:
                raise IndexError(
                    f"axis {axis} is out of bounds for array of dimension {values.ndim}"
                )

        repeat_counts = _pytorch_repeat_repeats(
            repeats,
            _np,
            _torch,
            device=values.device,
        )
        return _torch.repeat_interleave(values, repeat_counts, dim=dim)

    wrapped_repeat.__name__ = "repeat"
    wrapped_repeat.__doc__ = getattr(_np.repeat, "__doc__", None)
    wrapped_repeat._pyrecest_axis_contract = True
    setattr(backend, "repeat", wrapped_repeat)
    setattr(pytorch_backend, "repeat", wrapped_repeat)


def register_backend_submodules(backend: ModuleType | None = None) -> None:
    """Register virtual backend submodules for standard import statements."""
    if backend is None:
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel

    _adapt_cumulative_out_contract(backend)
    _adapt_pytorch_allclose_keyword_contract(backend)
    _adapt_pytorch_repeat_axis_contract(backend)

    backend.__path__ = getattr(backend, "__path__", [])
    backend_spec = getattr(backend, "__spec__", None)
    if backend_spec is not None:
        backend_spec.submodule_search_locations = (
            getattr(backend_spec, "submodule_search_locations", None) or []
        )

    for submodule_name in BACKEND_ATTRIBUTES:
        if not submodule_name:
            continue
        sys.modules[f"{backend.__name__}.{submodule_name}"] = getattr(
            backend, submodule_name
        )
