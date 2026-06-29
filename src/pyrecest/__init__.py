from importlib.metadata import PackageNotFoundError, version
from operator import index as _operator_index

import pyrecest._backend  # noqa
from pyrecest._backend_submodules import (  # noqa: F401
    register_backend_submodules as _register_backend_submodules,
)
from pyrecest.backend import copy  # noqa: F401

_register_backend_submodules()


def _patch_shared_numpy_copy_facade() -> None:
    """Make shared NumPy backend copy accept scalar and array-like inputs."""

    import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel

    if getattr(backend, "__backend_name__", None) not in {"autograd", "numpy"}:
        return

    original_copy = backend.copy

    def copy_arraylike(x):
        return original_copy(backend.array(x))

    copy_arraylike.__name__ = getattr(original_copy, "__name__", "copy")
    copy_arraylike.__doc__ = getattr(original_copy, "__doc__", None)
    backend.copy = copy_arraylike
    globals()["copy"] = backend.copy


def _patch_shared_numpy_squeeze_facade() -> None:
    """Make shared NumPy squeeze reject out-of-bounds axes before shape access."""

    import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel

    if getattr(backend, "__backend_name__", None) not in {"autograd", "numpy"}:
        return

    import pyrecest._backend._shared_numpy as shared_numpy  # pylint: disable=import-outside-toplevel

    original_squeeze = shared_numpy.squeeze
    np_module = shared_numpy._np

    def _axis_out_of_bounds_error(axis, ndim):
        axis_error = getattr(getattr(np_module, "exceptions", None), "AxisError", None)
        if axis_error is None:
            axis_error = getattr(np_module, "AxisError", None)
        if axis_error is None:
            return ValueError(
                f"axis {axis} is out of bounds for array of dimension {ndim}"
            )
        try:
            return axis_error(axis, ndim=ndim)
        except TypeError:  # pragma: no cover - compatibility with older NumPy APIs
            return axis_error(axis, ndim)

    def squeeze(x, axis=None):
        x_arr = np_module.asarray(x)
        if axis is None:
            return original_squeeze(x_arr, axis=None)

        axes = shared_numpy._normalize_squeeze_axes(axis)
        if not axes:
            return x_arr

        normalized_axes = []
        for one_axis in axes:
            if isinstance(one_axis, (int, np_module.integer)):
                one_axis = int(one_axis)
                normalized_axis = one_axis + x_arr.ndim if one_axis < 0 else one_axis
                if normalized_axis < 0 or normalized_axis >= x_arr.ndim:
                    raise _axis_out_of_bounds_error(one_axis, x_arr.ndim)
                normalized_axes.append(normalized_axis)
            else:
                normalized_axes.append(one_axis)
        normalized_axes = tuple(normalized_axes)

        if len(set(normalized_axes)) != len(normalized_axes):
            raise ValueError("duplicate value in 'axis'")
        if any(x_arr.shape[one_axis] != 1 for one_axis in normalized_axes):
            return x_arr
        squeeze_axis = (
            normalized_axes[0] if len(normalized_axes) == 1 else normalized_axes
        )
        return np_module.squeeze(x_arr, axis=squeeze_axis)

    squeeze.__name__ = getattr(original_squeeze, "__name__", "squeeze")
    squeeze.__doc__ = getattr(original_squeeze, "__doc__", None)
    shared_numpy.squeeze = squeeze
    backend.squeeze = squeeze


_patch_shared_numpy_copy_facade()
_patch_shared_numpy_squeeze_facade()


def _patch_pytorch_comparison_facade() -> None:
    """Make public PyTorch comparison helpers accept array-like inputs."""

    import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel

    if getattr(backend, "__backend_name__", None) != "pytorch":
        return

    try:
        import torch as _torch  # pylint: disable=import-outside-toplevel
    except (
        ModuleNotFoundError
    ):  # pragma: no cover - backend import fails first in practice
        return

    def _coerce_binary_args(x, y):
        device = next(
            (value.device for value in (x, y) if _torch.is_tensor(value)),
            None,
        )
        if not _torch.is_tensor(x):
            x = _torch.as_tensor(x, device=device)
        elif device is not None and x.device != device:
            x = x.to(device=device)
        if not _torch.is_tensor(y):
            y = _torch.as_tensor(y, device=device)
        elif device is not None and y.device != device:
            y = y.to(device=device)
        return x, y

    def _wrap_comparison(torch_func):
        def comparison(x, y, **kwargs):
            x, y = _coerce_binary_args(x, y)
            return torch_func(x, y, **kwargs)

        comparison.__name__ = getattr(torch_func, "__name__", "comparison")
        comparison.__doc__ = getattr(torch_func, "__doc__", None)
        return comparison

    backend.greater = _wrap_comparison(_torch.greater)
    backend.less = _wrap_comparison(_torch.less)
    backend.logical_or = _wrap_comparison(_torch.logical_or)


def _pytorch_tile_repetition(repetition) -> int:
    """Return one NumPy-style tile repetition as an integer."""

    try:
        return _operator_index(repetition)
    except TypeError as exc:
        raise TypeError("tile repetitions must be integers") from exc


def _pytorch_tile_repetitions(reps, numpy_module, torch_module) -> tuple[int, ...]:
    """Normalize NumPy-style tile repetitions for ``torch.Tensor.repeat``."""

    if torch_module.is_tensor(reps):
        reps = reps.detach().cpu().numpy()
    reps_array = numpy_module.asarray(reps)
    if reps_array.shape == ():
        repetitions = (_pytorch_tile_repetition(reps_array.item()),)
    else:
        repetitions = tuple(
            _pytorch_tile_repetition(one_repetition)
            for one_repetition in reps_array.tolist()
        )
    if any(one_repetition < 0 for one_repetition in repetitions):
        raise ValueError("negative dimensions are not allowed")
    return repetitions


def _patch_pytorch_tile_facade() -> None:
    """Make public PyTorch ``tile`` follow NumPy repetition semantics."""

    import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel

    if getattr(backend, "__backend_name__", None) != "pytorch":
        return

    try:
        import numpy as _np  # pylint: disable=import-outside-toplevel
        import torch as _torch  # pylint: disable=import-outside-toplevel
    except (
        ModuleNotFoundError
    ):  # pragma: no cover - backend import fails first in practice
        return

    def tile(x, reps):
        x = backend.array(x)
        repetitions = _pytorch_tile_repetitions(reps, _np, _torch)
        if not repetitions:
            return x.clone()
        if x.ndim < len(repetitions):
            x = x.reshape((1,) * (len(repetitions) - x.ndim) + tuple(x.shape))
        elif x.ndim > len(repetitions):
            repetitions = (1,) * (x.ndim - len(repetitions)) + repetitions
        return x.repeat(repetitions)

    tile.__name__ = "tile"
    tile.__doc__ = getattr(_np.tile, "__doc__", None)
    backend.tile = tile


def _pytorch_pad_pairs(pad_width, ndim, numpy_module) -> tuple[tuple[int, int], ...]:
    """Normalize NumPy-style pad widths to per-axis pairs."""

    try:
        pad_pairs = numpy_module.broadcast_to(numpy_module.asarray(pad_width), (ndim, 2))
    except ValueError as exc:
        raise ValueError(
            f"pad_width must be broadcastable to shape ({ndim}, 2)"
        ) from exc

    if numpy_module.any(pad_pairs < 0):
        raise ValueError("index can't contain negative values")

    return tuple(tuple(int(value) for value in pair) for pair in pad_pairs.tolist())


def _pytorch_torch_pad_width(pad_pairs: tuple[tuple[int, int], ...]) -> list[int]:
    """Convert NumPy-ordered pad pairs to PyTorch's reversed flat order."""

    return [value for pair in reversed(pad_pairs) for value in pair]


def _pytorch_constant_value_pairs(
    constant_values,
    ndim,
    numpy_module,
    torch_module,
) -> tuple[tuple[object, object], ...]:
    """Normalize NumPy-style constant pad values to per-axis pairs."""

    if torch_module.is_tensor(constant_values):
        constant_values = constant_values.detach().cpu().numpy()

    try:
        value_pairs = numpy_module.broadcast_to(
            numpy_module.asarray(constant_values),
            (ndim, 2),
        )
    except ValueError as exc:
        raise ValueError(
            f"constant_values must be broadcastable to shape ({ndim}, 2)"
        ) from exc

    return tuple(tuple(pair) for pair in value_pairs.tolist())


def _patch_pytorch_pad_facade() -> None:
    """Make public PyTorch ``pad`` accept NumPy-style constant values."""

    import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel

    if getattr(backend, "__backend_name__", None) != "pytorch":
        return

    try:
        import numpy as _np  # pylint: disable=import-outside-toplevel
        import torch as _torch  # pylint: disable=import-outside-toplevel
    except (
        ModuleNotFoundError
    ):  # pragma: no cover - backend import fails first in practice
        return

    original_pad = backend.pad

    def pad(a, pad_width, mode="constant", constant_values=0.0):
        values = backend.array(a)
        if mode != "constant":
            return original_pad(
                values,
                pad_width,
                mode=mode,
                constant_values=constant_values,
            )

        pad_pairs = _pytorch_pad_pairs(pad_width, values.ndim, _np)
        torch_pad_width = _pytorch_torch_pad_width(pad_pairs)
        result = _torch.nn.functional.pad(
            values,
            torch_pad_width,
            mode="constant",
            value=0.0,
        )
        value_pairs = _pytorch_constant_value_pairs(
            constant_values,
            values.ndim,
            _np,
            _torch,
        )

        for axis, ((before, after), (before_value, after_value)) in enumerate(
            zip(pad_pairs, value_pairs)
        ):
            if before:
                index = [slice(None)] * result.ndim
                index[axis] = slice(0, before)
                result[tuple(index)] = before_value
            if after:
                index = [slice(None)] * result.ndim
                index[axis] = slice(result.shape[axis] - after, result.shape[axis])
                result[tuple(index)] = after_value
        return result

    pad.__name__ = "pad"
    pad.__doc__ = getattr(original_pad, "__doc__", None)
    backend.pad = pad


def _patch_jax_std_out_facade() -> None:
    """Make public JAX ``std`` accept NumPy's ``out`` argument."""

    import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel

    if getattr(backend, "__backend_name__", None) != "jax":
        return

    original_std = backend.std

    def std(
        a, axis=None, dtype=None, out=None, ddof=0, keepdims=False, *, correction=0
    ):
        result = original_std(
            a,
            axis=axis,
            dtype=dtype,
            out=None,
            ddof=ddof,
            keepdims=keepdims,
            correction=correction,
        )
        if out is None:
            return result
        return backend.asarray(out).at[...].set(result)

    std.__name__ = getattr(original_std, "__name__", "std")
    std.__doc__ = getattr(original_std, "__doc__", None)
    backend.std = std


_patch_pytorch_comparison_facade()
_patch_pytorch_tile_facade()
_patch_pytorch_pad_facade()
_patch_jax_std_out_facade()

from pyrecest.backend_support import (  # noqa: E402,F401
    backend_support,
    format_backend_support_markdown,
    get_backend_support,
)
from pyrecest.backend_tools import (  # noqa: E402,F401
    assert_backend,
    get_backend_name,
    is_backend,
    warn_if_backend_env_changed,
)
from pyrecest.evidence import (  # noqa: E402,F401
    EvidenceComputationMode,
    resolve_evidence_computation_mode,
)
from pyrecest.exceptions import (  # noqa: E402,F401
    BackendNotSupportedError,
    BackendSupportError,
    DimensionMismatchError,
    NumericalStabilityError,
    OptionalDependencyError,
    PyRecEstError,
    ShapeError,
    ValidationError,
)

try:
    __version__ = version("pyrecest")
except PackageNotFoundError:  # pragma: no cover - editable/source tree without install metadata
    __version__ = "0+unknown"
