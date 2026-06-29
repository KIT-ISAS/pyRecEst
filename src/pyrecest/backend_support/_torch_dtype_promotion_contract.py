"""PyTorch dtype promotion compatibility helpers."""

from __future__ import annotations

from functools import wraps
from operator import index as _operator_index


def _copy_result_to_out(result, out, raw_pytorch):
    """Copy ``result`` into ``out`` and return the NumPy-style output object."""
    if out is None:
        return result
    copy_ = getattr(out, "copy_", None)
    if copy_ is not None:
        copy_(result)
        return out
    out[...] = raw_pytorch.to_numpy(result)
    return out


def _wrap_pytorch_unary_arraylike_special(original_func, raw_pytorch):
    """Return a unary PyTorch special-function wrapper accepting array-likes."""
    if getattr(original_func, "_pyrecest_arraylike_contract", False):
        return original_func

    @wraps(original_func)
    def special(a, *args, out=None, **kwargs):
        result = original_func(raw_pytorch.array(a), *args, **kwargs)
        return _copy_result_to_out(result, out, raw_pytorch)

    special._pyrecest_arraylike_contract = True
    return special


def _wrap_pytorch_gamma_arraylike(original_gamma, raw_pytorch, torch):
    """Return a PyTorch gamma wrapper accepting array-like inputs."""
    if getattr(original_gamma, "_pyrecest_arraylike_contract", False):
        return original_gamma

    @wraps(original_gamma)
    def gamma(a, out=None):
        result = torch.exp(raw_pytorch.gammaln(a))
        return _copy_result_to_out(result, out, raw_pytorch)

    gamma._pyrecest_arraylike_contract = True
    return gamma


def _wrap_pytorch_polygamma_arraylike(original_polygamma, raw_pytorch):
    """Return a PyTorch polygamma wrapper accepting array-like inputs."""
    if getattr(original_polygamma, "_pyrecest_arraylike_contract", False):
        return original_polygamma

    @wraps(original_polygamma)
    def polygamma(n, a, out=None):
        result = original_polygamma(_operator_index(n), raw_pytorch.array(a))
        return _copy_result_to_out(result, out, raw_pytorch)

    polygamma._pyrecest_arraylike_contract = True
    return polygamma


def _sync_active_pytorch_facade(raw_pytorch) -> None:
    """Keep the already-created public PyTorch facade aligned with raw patches."""
    try:
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - import fails before this module
        return
    if getattr(backend, "__backend_name__", None) != "pytorch":
        return
    for attribute_name in ("erf", "gammaln", "gamma", "polygamma"):
        setattr(backend, attribute_name, getattr(raw_pytorch, attribute_name))


def _patch_pytorch_special_arraylike_contract(raw_pytorch, torch) -> None:
    """Make PyTorch special helpers accept PyRecEst's array-like contract."""
    raw_pytorch.erf = _wrap_pytorch_unary_arraylike_special(raw_pytorch.erf, raw_pytorch)
    raw_pytorch.gammaln = _wrap_pytorch_unary_arraylike_special(
        raw_pytorch.gammaln,
        raw_pytorch,
    )
    raw_pytorch.gamma = _wrap_pytorch_gamma_arraylike(
        raw_pytorch.gamma,
        raw_pytorch,
        torch,
    )
    raw_pytorch.polygamma = _wrap_pytorch_polygamma_arraylike(
        raw_pytorch.polygamma,
        raw_pytorch,
    )
    _sync_active_pytorch_facade(raw_pytorch)


def _patch_pytorch_convert_to_wider_dtype(raw_pytorch, torch) -> None:
    """Make PyTorch mixed-dtype helpers use Torch's promotion rules."""
    original_convert = raw_pytorch.convert_to_wider_dtype
    if getattr(original_convert, "_pyrecest_torch_promotion_contract", False):
        return

    def convert_to_wider_dtype(tensor_list):
        tensors = list(tensor_list)
        if not tensors:
            return tensors

        promoted_dtype = tensors[0].dtype
        for tensor in tensors[1:]:
            promoted_dtype = torch.promote_types(promoted_dtype, tensor.dtype)

        if all(tensor.dtype == promoted_dtype for tensor in tensors):
            return tensors
        return [raw_pytorch.cast(tensor, dtype=promoted_dtype) for tensor in tensors]

    convert_to_wider_dtype.__name__ = getattr(
        original_convert, "__name__", "convert_to_wider_dtype"
    )
    convert_to_wider_dtype.__doc__ = getattr(original_convert, "__doc__", None)
    convert_to_wider_dtype._pyrecest_torch_promotion_contract = True
    raw_pytorch.convert_to_wider_dtype = convert_to_wider_dtype


def patch_pytorch_dtype_promotion_contract() -> None:
    """Patch PyTorch backend helpers that need NumPy-style facade contracts."""
    try:
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend import failed earlier
        return

    _patch_pytorch_convert_to_wider_dtype(raw_pytorch, torch)
    _patch_pytorch_special_arraylike_contract(raw_pytorch, torch)
