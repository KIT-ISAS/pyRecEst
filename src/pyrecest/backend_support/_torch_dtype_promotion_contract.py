"""PyTorch backend compatibility helpers."""

from __future__ import annotations


def _coerce_pytorch_binary_args(x, y, torch_module):
    """Return tensor operands on a common device for binary PyTorch helpers."""

    device = next(
        (value.device for value in (x, y) if torch_module.is_tensor(value)),
        None,
    )
    if not torch_module.is_tensor(x):
        x = torch_module.as_tensor(x, device=device)
    elif device is not None and x.device != device:
        x = x.to(device=device)

    if not torch_module.is_tensor(y):
        y = torch_module.as_tensor(y, device=device)
    elif device is not None and y.device != device:
        y = y.to(device=device)

    return x, y


def _wrap_pytorch_binary_helper(original_helper, torch_function, torch_module, name):
    """Wrap a binary PyTorch helper with NumPy-style array-like coercion."""

    if getattr(original_helper, "_pyrecest_arraylike_binary_contract", False):
        return original_helper

    def helper(x, y, **kwargs):
        x, y = _coerce_pytorch_binary_args(x, y, torch_module)
        return torch_function(x, y, **kwargs)

    helper.__name__ = getattr(original_helper, "__name__", name)
    helper.__doc__ = getattr(original_helper, "__doc__", None)
    helper._pyrecest_arraylike_binary_contract = True
    return helper


def _patch_pytorch_binary_arraylike_contract() -> None:
    """Make raw PyTorch binary helpers accept array-like operands."""

    try:
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend import failed earlier
        return

    active_pytorch_backend = getattr(backend, "__backend_name__", None) == "pytorch"
    for helper_name, torch_function in {
        "greater": torch.greater,
        "less": torch.less,
        "logical_or": torch.logical_or,
    }.items():
        helper = _wrap_pytorch_binary_helper(
            getattr(raw_pytorch, helper_name),
            torch_function,
            torch,
            helper_name,
        )
        setattr(raw_pytorch, helper_name, helper)
        if active_pytorch_backend:
            setattr(backend, helper_name, helper)


def patch_pytorch_dtype_promotion_contract() -> None:
    """Make PyTorch backend import-time compatibility shims active."""

    _patch_pytorch_binary_arraylike_contract()

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
