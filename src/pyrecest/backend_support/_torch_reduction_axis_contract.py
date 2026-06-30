"""PyTorch reduction-axis compatibility helpers."""

from __future__ import annotations

from operator import index as _operator_index


def _pytorch_axis_type_error():
    return TypeError("axis must be an integer or a sequence of integers")


def _is_pytorch_logical_axis(axis, torch_module, np):
    if isinstance(axis, (bool, np.bool_)):
        return True
    if isinstance(axis, np.ndarray):
        return axis.shape == () and axis.dtype.kind == "b"
    if torch_module.is_tensor(axis):
        return axis.ndim == 0 and axis.dtype == torch_module.bool
    return False


def _normalize_pytorch_axis_value(axis, torch_module, np):
    if _is_pytorch_logical_axis(axis, torch_module, np):
        raise _pytorch_axis_type_error()
    try:
        return _operator_index(axis)
    except TypeError as exc:
        raise _pytorch_axis_type_error() from exc


def _normalize_pytorch_axis(axis, torch_module, np):
    if axis is None:
        return None
    if isinstance(axis, (tuple, list)):
        return tuple(
            _normalize_pytorch_axis_value(one_axis, torch_module, np)
            for one_axis in axis
        )
    return _normalize_pytorch_axis_value(axis, torch_module, np)


def _normalize_pytorch_optional_axis(axis, dim, func_name, torch_module, np):
    if dim is None:
        return _normalize_pytorch_axis(axis, torch_module, np)

    dim = _normalize_pytorch_axis(dim, torch_module, np)
    if axis is not None and _normalize_pytorch_axis(axis, torch_module, np) != dim:
        raise TypeError(f"{func_name}() got both 'axis' and 'dim'")
    return dim


def _copy_metadata(wrapper, original, name):
    wrapper.__name__ = getattr(original, "__name__", name)
    wrapper.__doc__ = getattr(original, "__doc__", None)
    wrapper._pyrecest_axis_contract = True
    return wrapper


def _patch_mean(raw_pytorch, backend, torch, np, active_pytorch_backend):
    original_mean = raw_pytorch.mean
    if getattr(original_mean, "_pyrecest_axis_contract", False):
        return

    def mean(
        a,
        axis=None,
        dtype=None,
        out=None,
        keepdims=False,
        *,
        dim=None,
        keepdim=None,
    ):
        axis = _normalize_pytorch_optional_axis(axis, dim, "mean", torch, np)
        return original_mean(
            a,
            axis=axis,
            dtype=dtype,
            out=out,
            keepdims=keepdims,
            keepdim=keepdim,
        )

    raw_pytorch.mean = _copy_metadata(mean, original_mean, "mean")
    if active_pytorch_backend:
        backend.mean = raw_pytorch.mean


def _patch_std(raw_pytorch, backend, torch, np, active_pytorch_backend):
    original_std = raw_pytorch.std
    if getattr(original_std, "_pyrecest_axis_contract", False):
        return

    def std(
        a,
        axis=None,
        dtype=None,
        out=None,
        ddof=0,
        keepdims=False,
        *,
        correction=0,
        dim=None,
        keepdim=None,
    ):
        axis = _normalize_pytorch_optional_axis(axis, dim, "std", torch, np)
        return original_std(
            a,
            axis=axis,
            dtype=dtype,
            out=out,
            ddof=ddof,
            keepdims=keepdims,
            correction=correction,
            keepdim=keepdim,
        )

    raw_pytorch.std = _copy_metadata(std, original_std, "std")
    if active_pytorch_backend:
        backend.std = raw_pytorch.std


def _patch_sum(raw_pytorch, backend, torch, np, active_pytorch_backend):
    original_sum = raw_pytorch.sum
    if getattr(original_sum, "_pyrecest_axis_contract", False):
        return

    def sum(  # pylint: disable=redefined-builtin
        x,
        axis=None,
        keepdims=None,
        dtype=None,
        out=None,
        *,
        dim=None,
        keepdim=None,
    ):
        axis = _normalize_pytorch_optional_axis(axis, dim, "sum", torch, np)
        return original_sum(
            x,
            axis=axis,
            keepdims=keepdims,
            dtype=dtype,
            out=out,
            keepdim=keepdim,
        )

    raw_pytorch.sum = _copy_metadata(sum, original_sum, "sum")
    if active_pytorch_backend:
        backend.sum = raw_pytorch.sum


def _patch_quantile(raw_pytorch, backend, torch, np, active_pytorch_backend):
    original_quantile = raw_pytorch.quantile
    if getattr(original_quantile, "_pyrecest_axis_contract", False):
        return

    def quantile(
        a,
        q,
        axis=None,
        out=None,
        overwrite_input=False,
        method="linear",
        keepdims=False,
        *,
        dim=None,
        keepdim=None,
        interpolation=None,
    ):
        axis = _normalize_pytorch_optional_axis(axis, dim, "quantile", torch, np)
        return original_quantile(
            a,
            q,
            axis=axis,
            out=out,
            overwrite_input=overwrite_input,
            method=method,
            keepdims=keepdims,
            keepdim=keepdim,
            interpolation=interpolation,
        )

    raw_pytorch.quantile = _copy_metadata(quantile, original_quantile, "quantile")
    if active_pytorch_backend:
        backend.quantile = raw_pytorch.quantile


def _patch_arg_reduction(raw_pytorch, backend, torch, np, active_pytorch_backend, name):
    original = getattr(raw_pytorch, name)
    if getattr(original, "_pyrecest_axis_contract", False):
        return

    def arg_reduction(
        a,
        axis=None,
        out=None,
        keepdims=False,
        *,
        dim=None,
        keepdim=None,
    ):
        axis = _normalize_pytorch_optional_axis(axis, dim, name, torch, np)
        return original(
            a,
            axis=axis,
            out=out,
            keepdims=keepdims,
            keepdim=keepdim,
        )

    patched = _copy_metadata(arg_reduction, original, name)
    setattr(raw_pytorch, name, patched)
    if active_pytorch_backend:
        setattr(backend, name, patched)


def _patch_normalize_reduction_axes(raw_pytorch, torch, np):
    original_normalize_axes = getattr(raw_pytorch, "_normalize_reduction_axes", None)
    if original_normalize_axes is None or getattr(
        original_normalize_axes,
        "_pyrecest_axis_contract",
        False,
    ):
        return

    def _normalize_reduction_axes(axis, ndim_):
        axis = _normalize_pytorch_axis(axis, torch, np)
        axes = (axis,) if isinstance(axis, int) else tuple(axis)

        normalized_axes = tuple(
            one_axis + ndim_ if one_axis < 0 else one_axis for one_axis in axes
        )
        if len(set(normalized_axes)) != len(normalized_axes):
            raise ValueError("duplicate value in 'axis'")

        for one_axis, normalized_axis in zip(axes, normalized_axes):
            if normalized_axis < 0 or normalized_axis >= ndim_:
                raise IndexError(
                    f"axis {one_axis} is out of bounds for array of dimension {ndim_}"
                )

        return normalized_axes

    _normalize_reduction_axes.__name__ = "_normalize_reduction_axes"
    _normalize_reduction_axes.__doc__ = getattr(
        original_normalize_axes,
        "__doc__",
        None,
    )
    _normalize_reduction_axes._pyrecest_axis_contract = True
    raw_pytorch._normalize_reduction_axes = _normalize_reduction_axes


def patch_pytorch_reduction_axis_contract(raw_pytorch, torch, np) -> None:
    """Normalize scalar array/tensor axes before PyTorch reduction dispatch."""
    try:
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - import fails before this module
        backend = None

    active_pytorch_backend = (
        backend is not None and getattr(backend, "__backend_name__", None) == "pytorch"
    )

    _patch_mean(raw_pytorch, backend, torch, np, active_pytorch_backend)
    _patch_std(raw_pytorch, backend, torch, np, active_pytorch_backend)
    _patch_sum(raw_pytorch, backend, torch, np, active_pytorch_backend)
    _patch_quantile(raw_pytorch, backend, torch, np, active_pytorch_backend)
    _patch_arg_reduction(
        raw_pytorch, backend, torch, np, active_pytorch_backend, "argmax"
    )
    _patch_arg_reduction(
        raw_pytorch, backend, torch, np, active_pytorch_backend, "argmin"
    )
    _patch_normalize_reduction_axes(raw_pytorch, torch, np)
