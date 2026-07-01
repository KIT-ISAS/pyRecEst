"""Runtime metadata helpers for public API stability."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from typing import Final, Literal, ParamSpec, TypeVar

from pyrecest.backend_support._pytorch_allclose_device_contract import (
    patch_pytorch_allclose_device_contract as _patch_pytorch_allclose_device_contract,
)


def _patch_pytorch_triangular_vector_helpers() -> None:
    """Make PyTorch triangular-vector helpers accept array-like inputs."""

    try:
        import pyrecest._backend.pytorch as pytorch_backend  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
        import torch as torch_module  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend may be unavailable
        return

    originals = {
        "vec_to_diag": getattr(pytorch_backend, "vec_to_diag", None),
        "tril_to_vec": getattr(pytorch_backend, "tril_to_vec", None),
        "triu_to_vec": getattr(pytorch_backend, "triu_to_vec", None),
    }
    if any(original is None for original in originals.values()):
        return

    if all(
        getattr(original, "_pyrecest_triangular_vector_array_like_contract", False)
        for original in originals.values()
    ):
        if getattr(backend, "__backend_name__", None) == "pytorch":
            for name, helper in originals.items():
                setattr(backend, name, helper)
        return

    def vec_to_diag(vec):
        values = pytorch_backend.array(vec)
        return torch_module.diag_embed(values, offset=0)

    def _triangular_to_vec(x, k, indices_func):
        values = pytorch_backend.array(x)
        rows, cols = indices_func(values.shape[-1], k=k)
        rows = rows.to(device=values.device)
        cols = cols.to(device=values.device)
        return values[..., rows, cols]

    def tril_to_vec(x, k=0):
        return _triangular_to_vec(x, k, pytorch_backend.tril_indices)

    def triu_to_vec(x, k=0):
        return _triangular_to_vec(x, k, pytorch_backend.triu_indices)

    patched = {
        "vec_to_diag": vec_to_diag,
        "tril_to_vec": tril_to_vec,
        "triu_to_vec": triu_to_vec,
    }
    for name, helper in patched.items():
        original = originals[name]
        helper.__name__ = getattr(original, "__name__", name)
        helper.__doc__ = getattr(original, "__doc__", None)
        helper._pyrecest_triangular_vector_array_like_contract = True
        setattr(pytorch_backend, name, helper)
        if getattr(backend, "__backend_name__", None) == "pytorch":
            setattr(backend, name, helper)


_patch_pytorch_allclose_device_contract()
_patch_pytorch_triangular_vector_helpers()

P = ParamSpec("P")
R = TypeVar("R")

StabilityLevel = Literal[
    "stable", "experimental", "deprecated", "backend-specific", "internal"
]
STABILITY_LEVELS: Final = (
    "stable",
    "experimental",
    "deprecated",
    "backend-specific",
    "internal",
)


@dataclass(frozen=True)
class PublicAPIStatus:
    """Stability metadata for a public API entry."""

    name: str
    level: StabilityLevel
    since: str | None = None
    remove_in: str | None = None
    replacement: str | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if self.level not in STABILITY_LEVELS:
            raise ValueError(f"Unknown stability level: {self.level!r}")

    def to_dict(self) -> dict[str, str | None]:
        """Return a JSON-serializable representation."""
        return asdict(self)


_PUBLIC_API_STATUS: Final[dict[str, PublicAPIStatus]] = {
    "KalmanFilter": PublicAPIStatus(
        "KalmanFilter", "stable", since="2.2.0", notes="Core linear Gaussian filter."
    ),
    "GaussianDistribution": PublicAPIStatus(
        "GaussianDistribution",
        "stable",
        since="2.2.0",
        notes="Core Euclidean distribution.",
    ),
    "pyrecest.backend": PublicAPIStatus(
        "pyrecest.backend",
        "backend-specific",
        since="2.2.0",
        notes="Support depends on the backend capability matrix.",
    ),
    "UKFOnManifolds": PublicAPIStatus(
        "UKFOnManifolds",
        "backend-specific",
        since="2.2.0",
        notes="Backend exclusions are documented in the backend API matrix.",
    ),
}


def stability(
    level: StabilityLevel,
    *,
    since: str | None = None,
    remove_in: str | None = None,
    replacement: str | None = None,
    notes: str = "",
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Attach stability metadata to a function, method, or class."""
    if level not in STABILITY_LEVELS:
        raise ValueError(f"Unknown stability level: {level!r}")

    def decorator(obj: Callable[P, R]) -> Callable[P, R]:
        status = PublicAPIStatus(
            name=f"{obj.__module__}.{obj.__qualname__}",
            level=level,
            since=since,
            remove_in=remove_in,
            replacement=replacement,
            notes=notes,
        )
        setattr(obj, "__pyrecest_stability__", status)
        return obj

    return decorator


def get_public_api_status(name: str) -> PublicAPIStatus | None:
    """Return registered stability metadata for a public API name."""
    return _PUBLIC_API_STATUS.get(name)


def iter_public_api_status() -> Iterable[PublicAPIStatus]:
    """Iterate registered public API stability rows in stable name order."""
    return tuple(_PUBLIC_API_STATUS[name] for name in sorted(_PUBLIC_API_STATUS))


__all__ = [
    "PublicAPIStatus",
    "STABILITY_LEVELS",
    "StabilityLevel",
    "get_public_api_status",
    "iter_public_api_status",
    "stability",
]
