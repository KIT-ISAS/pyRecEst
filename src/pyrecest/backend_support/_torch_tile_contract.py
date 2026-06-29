"""NumPy-compatible tile helper for the PyTorch backend."""

from __future__ import annotations

from operator import index as _index


def _one_count(value) -> int:
    try:
        return _index(value)
    except TypeError as exc:
        raise TypeError("tile repetitions must be integers") from exc


def _counts(reps, np_module, torch_module) -> tuple[int, ...]:
    if torch_module.is_tensor(reps):
        reps = reps.detach().cpu().numpy()
    reps_array = np_module.asarray(reps)
    if reps_array.shape == ():
        result = (_one_count(reps_array.item()),)
    else:
        result = tuple(_one_count(value) for value in reps_array.tolist())
    if any(value < 0 for value in result):
        raise ValueError("negative dimensions are not allowed")
    return result


def install() -> None:
    """Install NumPy-compatible tile semantics for the PyTorch backend module."""
    try:
        import numpy as np_module  # pylint: disable=import-outside-toplevel
        import pyrecest._backend.pytorch as torch_backend  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as public_backend  # pylint: disable=import-outside-toplevel
        import torch as torch_module  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - optional PyTorch backend
        return

    current = getattr(torch_backend, "tile", None)
    if current is None or getattr(current, "_pyrecest_numpy_contract", False):
        return

    def tile(x, reps):
        x = torch_backend.array(x)
        reps_tuple = _counts(reps, np_module, torch_module)
        if not reps_tuple:
            return x.clone()
        if x.ndim < len(reps_tuple):
            x = x.reshape((1,) * (len(reps_tuple) - x.ndim) + tuple(x.shape))
        elif x.ndim > len(reps_tuple):
            reps_tuple = (1,) * (x.ndim - len(reps_tuple)) + reps_tuple
        return x.repeat(reps_tuple)

    tile.__name__ = getattr(current, "__name__", "tile")
    tile.__doc__ = getattr(np_module.tile, "__doc__", None)
    tile._pyrecest_numpy_contract = True
    torch_backend.tile = tile
    if getattr(public_backend, "__backend_name__", None) == "pytorch":
        public_backend.tile = tile
