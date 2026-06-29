"""PyTorch dtype promotion compatibility helpers."""

from __future__ import annotations


def _return_or_store_predicate_out(raw_pytorch, result, out):
    """Return ``result`` or store it through a NumPy/PyTorch-style ``out``."""

    if out is None:
        return result
    copy_ = getattr(out, "copy_", None)
    if copy_ is not None:
        copy_(result)
        return out
    out[...] = raw_pytorch.to_numpy(result)
    return out


def _patch_pytorch_predicate_numpy_contract(raw_pytorch, torch, backend) -> None:
    """Make PyTorch predicate helpers accept NumPy-style array-like inputs."""

    active_pytorch_backend = getattr(backend, "__backend_name__", None) == "pytorch"

    def _make_predicate(helper_name, torch_func, original_helper):
        def predicate(a, out=None):
            result = torch_func(raw_pytorch.array(a))
            return _return_or_store_predicate_out(raw_pytorch, result, out)

        predicate.__name__ = getattr(original_helper, "__name__", helper_name)
        predicate.__doc__ = getattr(original_helper, "__doc__", None)
        predicate._pyrecest_numpy_contract = True
        return predicate

    for helper_name in ("isfinite", "isinf", "isnan", "isreal"):
        original_helper = getattr(raw_pytorch, helper_name)
        if getattr(original_helper, "_pyrecest_numpy_contract", False):
            helper = original_helper
        else:
            helper = _make_predicate(
                helper_name,
                getattr(torch, helper_name),
                original_helper,
            )
            setattr(raw_pytorch, helper_name, helper)
        if active_pytorch_backend:
            setattr(backend, helper_name, helper)


def patch_pytorch_dtype_promotion_contract() -> None:
    """Make PyTorch mixed-dtype helpers use Torch's promotion rules."""
    try:
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend import failed earlier
        return

    original_convert = raw_pytorch.convert_to_wider_dtype
    if not getattr(original_convert, "_pyrecest_torch_promotion_contract", False):

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

    _patch_pytorch_predicate_numpy_contract(raw_pytorch, torch, backend)
