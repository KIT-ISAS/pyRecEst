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

    missing_value_key = "_".join(("equal", "nan"))

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
        if repeats.dtype.is_floating_point or repeats.dtype.is_complex:
            raise TypeError("repeat counts must be integers")
        repeat_counts = repeats.to(device=device, dtype=torch_module.long)
        if bool(torch_module.any(repeat_counts < 0)):
            raise ValueError("repeats may not contain negative values")
        return repeat_counts

    repeats_array = numpy_module.asarray(repeats)
    if repeats_array.shape == ():
        return _pytorch_repeat_count(repeats_array.item())
    if not numpy_module.can_cast(
        repeats_array.dtype, numpy_module.dtype("intp"), casting="safe"
    ):
        raise TypeError("repeat counts must be integers")
    repeat_counts = torch_module.as_tensor(
        repeats_array, dtype=torch_module.long, device=device
    )
    if bool(torch_module.any(repeat_counts < 0)):
        raise ValueError("repeats may not contain negative values")
    return repeat_counts


def _pytorch_repeat_with_arraylike_inputs(
    repeat_interleave, array_func, numpy_module, torch_module
):
    """Return a NumPy-compatible ``repeat`` wrapper for the PyTorch backend."""

    @wraps(repeat_interleave)
    def repeat(a, repeats, axis=None, *, dim=None, output_size=None):
        if dim is not None:
            if axis is not None and axis != dim:
                raise TypeError("repeat() got both 'axis' and 'dim'")
            axis = dim
        if axis is not None:
            axis = _operator_index(axis)

        a = array_func(a)
        repeat_counts = _pytorch_repeat_counts(
            repeats,
            numpy_module=numpy_module,
            torch_module=torch_module,
            device=a.device,
        )
        kwargs = {"dim": axis}
        if output_size is not None:
            kwargs["output_size"] = output_size
        return repeat_interleave(a, repeat_counts, **kwargs)

    repeat._pyrecest_repeat_contract = True
    return repeat


def _adapt_pytorch_repeat_contract(backend: ModuleType) -> None:
    """Adapt PyTorch ``repeat`` to PyRecEst's NumPy-style backend contract."""
    if getattr(backend, "__backend_name__", None) != "pytorch":
        return

    import numpy as numpy_module  # pylint: disable=import-outside-toplevel
    import pyrecest._backend.pytorch as pytorch_backend  # pylint: disable=import-outside-toplevel
    import torch as torch_module  # pylint: disable=import-outside-toplevel

    repeat = getattr(pytorch_backend, "repeat", None)
    if repeat is None or getattr(repeat, "_pyrecest_repeat_contract", False):
        return

    wrapped_repeat = _pytorch_repeat_with_arraylike_inputs(
        repeat,
        backend.array,
        numpy_module,
        torch_module,
    )
    setattr(pytorch_backend, "repeat", wrapped_repeat)
    setattr(backend, "repeat", wrapped_repeat)


def _pytorch_reshape_shape(shape, torch_module) -> tuple[int, ...]:
    """Normalize NumPy-style reshape dimensions for ``torch.reshape``."""
    if torch_module.is_tensor(shape):
        if shape.ndim == 0:
            return (_operator_index(shape.item()),)
        shape = shape.detach().cpu().tolist()
    elif getattr(shape, "ndim", None) == 0 and hasattr(shape, "item"):
        return (_operator_index(shape.item()),)

    try:
        return (_operator_index(shape),)
    except TypeError:
        pass

    if isinstance(shape, (str, bytes)):
        raise TypeError("reshape shape must be an integer or a sequence of integers")

    try:
        return tuple(_operator_index(dimension) for dimension in shape)
    except TypeError as exc:
        raise TypeError(
            "reshape shape must be an integer or a sequence of integers"
        ) from exc


def _adapt_pytorch_reshape_contract(backend: ModuleType) -> None:
    """Adapt PyTorch reshape to accept array-like inputs and NumPy-style shapes."""
    if getattr(backend, "__backend_name__", None) != "pytorch":
        return

    import pyrecest._backend.pytorch as pytorch_backend  # pylint: disable=import-outside-toplevel
    import torch as torch_module  # pylint: disable=import-outside-toplevel

    reshape = getattr(pytorch_backend, "reshape", None)
    if reshape is None or getattr(reshape, "_pyrecest_reshape_contract", False):
        return

    @wraps(reshape)
    def wrapped_reshape(x, shape):
        return reshape(pytorch_backend.array(x), _pytorch_reshape_shape(shape, torch_module))

    wrapped_reshape._pyrecest_reshape_contract = True
    setattr(pytorch_backend, "reshape", wrapped_reshape)
    setattr(backend, "reshape", wrapped_reshape)


def register_backend_submodules(backend: ModuleType | None = None) -> None:
    """Register virtual backend submodules for standard import statements."""
    if backend is None:
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel

    _adapt_cumulative_out_contract(backend)
    _adapt_pytorch_allclose_keyword_contract(backend)
    _adapt_pytorch_repeat_contract(backend)
    _adapt_pytorch_reshape_contract(backend)

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
