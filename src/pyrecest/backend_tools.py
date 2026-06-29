"""Small helpers for inspecting the process-global PyRecEst backend."""

from __future__ import annotations

import os
import warnings
from operator import index as _operator_index


def _pytorch_scalar_tensor_index(index, torch_module):
    """Return Python int indices for scalar integer tensors."""

    if not torch_module.is_tensor(index) or index.ndim != 0:
        return index
    if (
        index.dtype in {torch_module.bool, torch_module.uint8}
        or index.dtype.is_floating_point
        or index.dtype.is_complex
    ):
        return index
    return _operator_index(index)


def _wrap_pytorch_assignment_helper(original_assignment, torch_module):
    """Normalize scalar tensor indices before assignment helper len() checks."""

    def assignment(x, values, indices, axis=0):
        indices = _pytorch_scalar_tensor_index(indices, torch_module)
        return original_assignment(x, values, indices, axis=axis)

    assignment.__name__ = getattr(original_assignment, "__name__", "assignment")
    assignment.__doc__ = getattr(original_assignment, "__doc__", None)
    return assignment


def _patch_pytorch_assignment_scalar_tensor_indices() -> None:
    """Make PyTorch assignment helpers accept scalar integer tensor indices."""

    try:
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - import fails before this module
        return

    if getattr(backend, "__backend_name__", None) != "pytorch":
        return

    try:
        import pyrecest._backend.pytorch as pytorch_backend  # pylint: disable=import-outside-toplevel
        import torch as _torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend import failed earlier
        return

    backend.assignment = _wrap_pytorch_assignment_helper(backend.assignment, _torch)
    backend.assignment_by_sum = _wrap_pytorch_assignment_helper(
        backend.assignment_by_sum, _torch
    )
    pytorch_backend.assignment = _wrap_pytorch_assignment_helper(
        pytorch_backend.assignment, _torch
    )
    pytorch_backend.assignment_by_sum = _wrap_pytorch_assignment_helper(
        pytorch_backend.assignment_by_sum, _torch
    )


def _patch_pytorch_diag_numpy_contract() -> None:
    """Make PyTorch diag accept array-like inputs and NumPy's ``k`` keyword."""

    try:
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - import fails before this module
        return

    if getattr(backend, "__backend_name__", None) != "pytorch":
        return

    try:
        import pyrecest._backend.pytorch as pytorch_backend  # pylint: disable=import-outside-toplevel
        import torch as _torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend import failed earlier
        return

    if getattr(pytorch_backend.diag, "_pyrecest_numpy_contract", False):
        return

    def diag(v, k=0):
        return _torch.diag(pytorch_backend.array(v), diagonal=k)

    diag.__name__ = getattr(_torch.diag, "__name__", "diag")
    diag.__doc__ = getattr(_torch.diag, "__doc__", None)
    diag._pyrecest_numpy_contract = True
    backend.diag = diag
    pytorch_backend.diag = diag


def _patch_pytorch_broadcast_arrays_numpy_contract() -> None:
    """Make PyTorch broadcast_arrays accept NumPy-style array-like inputs."""

    try:
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - import fails before this module
        return

    if getattr(backend, "__backend_name__", None) != "pytorch":
        return

    try:
        import pyrecest._backend.pytorch as pytorch_backend  # pylint: disable=import-outside-toplevel
        import torch as _torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend import failed earlier
        return

    if getattr(pytorch_backend.broadcast_arrays, "_pyrecest_numpy_contract", False):
        return

    def broadcast_arrays(*arrays):
        tensors = tuple(pytorch_backend.array(array) for array in arrays)
        return _torch.broadcast_tensors(*tensors)

    broadcast_arrays.__name__ = "broadcast_arrays"
    broadcast_arrays.__doc__ = getattr(_torch.broadcast_tensors, "__doc__", None)
    broadcast_arrays._pyrecest_numpy_contract = True
    backend.broadcast_arrays = broadcast_arrays
    pytorch_backend.broadcast_arrays = broadcast_arrays


def _patch_pytorch_round_numpy_contract() -> None:
    """Make PyTorch round accept NumPy-style array-like inputs."""

    try:
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - import fails before this module
        return

    if getattr(backend, "__backend_name__", None) != "pytorch":
        return

    try:
        import pyrecest._backend.pytorch as pytorch_backend  # pylint: disable=import-outside-toplevel
        import torch as _torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend import failed earlier
        return

    if getattr(pytorch_backend.round, "_pyrecest_numpy_contract", False):
        return

    def round(a, decimals=0, out=None):  # pylint: disable=redefined-builtin
        decimals = _operator_index(decimals)
        result = _torch.round(pytorch_backend.array(a), decimals=decimals)
        if out is not None:
            out.copy_(result)
            return out
        return result

    round.__name__ = getattr(_torch.round, "__name__", "round")
    round.__doc__ = getattr(_torch.round, "__doc__", None)
    round._pyrecest_numpy_contract = True
    backend.round = round
    pytorch_backend.round = round


def _patch_pytorch_special_numpy_contract() -> None:
    """Make PyTorch special functions accept NumPy-style array-like inputs."""

    try:
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - import fails before this module
        return

    if getattr(backend, "__backend_name__", None) != "pytorch":
        return

    try:
        import pyrecest._backend.pytorch as pytorch_backend  # pylint: disable=import-outside-toplevel
        import torch as _torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend import failed earlier
        return

    def _return_or_store_out(result, out):
        if out is not None:
            out.copy_(result)
            return out
        return result

    def erf(a, out=None):
        result = _torch.erf(pytorch_backend.array(a))
        return _return_or_store_out(result, out)

    def gammaln(a, out=None):
        result = _torch.special.gammaln(pytorch_backend.array(a))
        return _return_or_store_out(result, out)

    def gamma(a, out=None):
        result = _torch.exp(gammaln(a))
        return _return_or_store_out(result, out)

    def polygamma(n, a, out=None):
        result = _torch.polygamma(n, pytorch_backend.array(a))
        return _return_or_store_out(result, out)

    for name, helper, target in (
        ("erf", erf, _torch.erf),
        ("gammaln", gammaln, _torch.special.gammaln),
        ("gamma", gamma, pytorch_backend.gamma),
        ("polygamma", polygamma, _torch.polygamma),
    ):
        helper.__name__ = name
        helper.__doc__ = getattr(target, "__doc__", None)
        helper._pyrecest_numpy_contract = True
        setattr(backend, name, helper)
        setattr(pytorch_backend, name, helper)

    pytorch_backend._gammaln = gammaln  # pylint: disable=protected-access


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


def _patch_raw_pytorch_tile_numpy_contract() -> None:
    """Make raw PyTorch tile follow NumPy repetition semantics."""

    try:
        import numpy as _np  # pylint: disable=import-outside-toplevel
        from pyrecest._backend import (  # pylint: disable=import-outside-toplevel
            pytorch as pytorch_backend,
        )
        import torch as _torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend not installed
        return

    if getattr(pytorch_backend.tile, "_pyrecest_numpy_contract", False):
        return

    def tile(x, reps):
        x = pytorch_backend.array(x)
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
    tile._pyrecest_numpy_contract = True
    pytorch_backend.tile = tile

    try:
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - import fails before this module
        return
    if getattr(backend, "__backend_name__", None) == "pytorch":
        backend.tile = tile


_patch_pytorch_assignment_scalar_tensor_indices()
_patch_pytorch_diag_numpy_contract()
_patch_pytorch_broadcast_arrays_numpy_contract()
_patch_pytorch_round_numpy_contract()
_patch_pytorch_special_numpy_contract()
_patch_raw_pytorch_tile_numpy_contract()


def get_backend_name() -> str:
    """Return the backend selected at import time."""
    import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel

    return backend.__backend_name__  # pylint: disable=no-member


def is_backend(expected: str | tuple[str, ...]) -> bool:
    """Return whether the active backend matches one of the expected names."""
    expected_names = _normalize_expected_backend_names(expected)
    return get_backend_name() in expected_names


def _normalize_expected_backend_names(
    expected: str | tuple[str, ...],
) -> tuple[str, ...]:
    message = "expected must name at least one backend."
    if isinstance(expected, str):
        names = (expected,)
    else:
        try:
            names = tuple(expected)
        except TypeError as exc:
            raise ValueError(message) from exc
    if not names or any(
        not isinstance(name, str) or not name or name.strip() != name for name in names
    ):
        raise ValueError(message)
    return tuple(dict.fromkeys(names))


def assert_backend(expected: str | tuple[str, ...]) -> None:
    """Raise ``RuntimeError`` unless the active backend matches ``expected``.

    Parameters
    ----------
    expected : str or tuple[str, ...]
        Allowed backend name or names.
    """
    active = get_backend_name()
    expected_names = _normalize_expected_backend_names(expected)
    if active not in expected_names:
        allowed = ", ".join(expected_names)
        raise RuntimeError(
            f"Expected PyRecEst backend {allowed}; active backend is {active}."
        )


def warn_if_backend_env_changed() -> None:
    """Warn when ``PYRECEST_BACKEND`` no longer matches the imported backend.

    Backend selection is process-global and import-time only. Changing the
    environment variable after importing :mod:`pyrecest` does not switch the
    already constructed backend facade.
    """
    active = get_backend_name()
    requested = os.environ.get("PYRECEST_BACKEND", active)
    if requested != active:
        warnings.warn(
            "PYRECEST_BACKEND was changed after pyrecest was imported. "
            f"The active backend remains {active!r}; the environment now requests "
            f"{requested!r}. Start a new Python process to switch backends.",
            RuntimeWarning,
            stacklevel=2,
        )
