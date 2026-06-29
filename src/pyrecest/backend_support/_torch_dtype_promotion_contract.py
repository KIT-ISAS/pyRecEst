"""PyTorch dtype promotion compatibility helpers."""

from __future__ import annotations


def _wrap_get_slice_arraylike_contract(original_get_slice, array_func, is_array_func=None):
    """Return a get_slice wrapper that accepts documented array-like inputs."""

    if getattr(original_get_slice, "_pyrecest_arraylike_contract", False):
        return original_get_slice

    def get_slice(x, indices):
        if is_array_func is None or not is_array_func(x):
            x = array_func(x)
        return original_get_slice(x, indices)

    get_slice.__name__ = getattr(original_get_slice, "__name__", "get_slice")
    get_slice.__doc__ = getattr(original_get_slice, "__doc__", None)
    get_slice._pyrecest_arraylike_contract = True
    return get_slice


def _patch_one_get_slice_module(module) -> None:
    original_get_slice = getattr(module, "get_slice", None)
    array_func = getattr(module, "array", None)
    if original_get_slice is None or array_func is None:
        return

    module.get_slice = _wrap_get_slice_arraylike_contract(
        original_get_slice,
        array_func,
        getattr(module, "is_array", None),
    )


def _patch_backend_get_slice_arraylike_contract() -> None:
    """Make public and raw backend get_slice helpers accept array-like inputs."""

    try:
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - import fails before this module
        backend = None

    if backend is not None:
        _patch_one_get_slice_module(backend)

    try:
        import pyrecest._backend._shared_numpy as shared_numpy  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - unavailable backend module
        pass
    else:
        _patch_one_get_slice_module(shared_numpy)

    try:
        import pyrecest._backend.numpy as raw_numpy  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - unavailable backend module
        pass
    else:
        _patch_one_get_slice_module(raw_numpy)

    try:
        import pyrecest._backend.autograd as raw_autograd  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - optional backend unavailable
        pass
    else:
        _patch_one_get_slice_module(raw_autograd)

    try:
        import pyrecest._backend.jax as raw_jax  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - optional backend unavailable
        pass
    else:
        _patch_one_get_slice_module(raw_jax)

    try:
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - optional backend unavailable
        pass
    else:
        _patch_one_get_slice_module(raw_pytorch)


def patch_pytorch_dtype_promotion_contract() -> None:
    """Make PyTorch mixed-dtype helpers use Torch's promotion rules."""
    try:
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend import failed earlier
        return

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


_patch_backend_get_slice_arraylike_contract()
