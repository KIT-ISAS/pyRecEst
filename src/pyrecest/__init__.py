from importlib.metadata import PackageNotFoundError, version

import pyrecest._backend  # noqa
from pyrecest._backend_submodules import (  # noqa: F401
    register_backend_submodules as _register_backend_submodules,
)
from pyrecest.backend import copy  # noqa: F401

_register_backend_submodules()


def _patch_pytorch_logical_or_arraylike():
    """Make the public PyTorch backend logical_or accept array-like inputs."""
    import pyrecest.backend as _backend  # pylint: disable=import-outside-toplevel

    if getattr(_backend, "__backend_name__", None) != "pytorch":
        return

    try:
        import torch as _torch  # pylint: disable=import-error,import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - inconsistent PyTorch backend env
        return

    def logical_or(x, y):
        device = None
        if _torch.is_tensor(x):
            device = x.device
        elif _torch.is_tensor(y):
            device = y.device
        return _torch.logical_or(
            _torch.as_tensor(x, device=device),
            _torch.as_tensor(y, device=device),
        )

    _backend.logical_or = logical_or


_patch_pytorch_logical_or_arraylike()

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
