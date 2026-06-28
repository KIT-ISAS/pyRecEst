from importlib.metadata import PackageNotFoundError, version

import pyrecest._backend  # noqa
from pyrecest._backend_submodules import (  # noqa: F401
    register_backend_submodules as _register_backend_submodules,
)
from pyrecest.backend import copy  # noqa: F401

_register_backend_submodules()


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


def _pytorch_tile_repetitions(reps, numpy_module, torch_module) -> tuple[int, ...]:
    """Normalize NumPy-style tile repetitions for ``torch.Tensor.repeat``."""

    if torch_module.is_tensor(reps):
        reps = reps.detach().cpu().numpy()
    reps_array = numpy_module.asarray(reps)
    if reps_array.shape == ():
        repetitions = (int(reps_array.item()),)
    else:
        repetitions = tuple(
            int(one_repetition) for one_repetition in reps_array.tolist()
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


_patch_pytorch_comparison_facade()
_patch_pytorch_tile_facade()

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
from pyrecest.stability import (  # noqa: E402,F401
    get_public_api_status,
    iter_public_api_status,
    stability,
)

try:
    __version__ = version("pyrecest")
except PackageNotFoundError:  # pragma: no cover - source tree without install metadata
    __version__ = "0+unknown"

__all__ = [
    "BackendNotSupportedError",
    "BackendSupportError",
    "DimensionMismatchError",
    "EvidenceComputationMode",
    "NumericalStabilityError",
    "OptionalDependencyError",
    "PyRecEstError",
    "ShapeError",
    "ValidationError",
    "__version__",
    "assert_backend",
    "backend_support",
    "copy",
    "format_backend_support_markdown",
    "get_backend_name",
    "get_backend_support",
    "get_public_api_status",
    "is_backend",
    "iter_public_api_status",
    "stability",
    "warn_if_backend_env_changed",
    "resolve_evidence_computation_mode",
]
