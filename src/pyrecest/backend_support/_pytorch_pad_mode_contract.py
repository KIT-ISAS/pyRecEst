"""PyTorch ``pad`` NumPy mode-name compatibility hook."""

from __future__ import annotations

_PAD_MODE_ALIASES = {
    "edge": "replicate",
    "wrap": "circular",
}


def _torch_pad_mode(mode):
    """Return the PyTorch padding mode corresponding to a NumPy mode name."""

    return _PAD_MODE_ALIASES.get(mode, mode)


def patch_pytorch_pad_mode_contract() -> None:
    """Patch raw/public PyTorch ``pad`` to accept NumPy mode names."""

    try:
        import pyrecest._backend.pytorch as pytorch_backend  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend may be unavailable
        return

    original_pad = getattr(pytorch_backend, "pad", None)
    if original_pad is None:
        return
    if getattr(original_pad, "_pyrecest_numpy_pad_mode_contract", False):
        if getattr(backend, "__backend_name__", None) == "pytorch":
            backend.pad = original_pad
        return

    def pad(a, pad_width, mode="constant", constant_values=0.0):
        return original_pad(
            a,
            pad_width,
            mode=_torch_pad_mode(mode),
            constant_values=constant_values,
        )

    pad.__name__ = getattr(original_pad, "__name__", "pad")
    pad.__doc__ = getattr(original_pad, "__doc__", None)
    pad._pyrecest_numpy_pad_mode_contract = True
    pytorch_backend.pad = pad
    if getattr(backend, "__backend_name__", None) == "pytorch":
        backend.pad = pad
