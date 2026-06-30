"""PyTorch dtype promotion compatibility helpers."""

from __future__ import annotations


def _patch_pytorch_randint_dtype_contract(raw_pytorch, torch) -> None:
    """Make PyTorch randint reject non-integer result dtypes."""
    pytorch_random = getattr(raw_pytorch, "random", None)
    if pytorch_random is None:
        return

    original_randint = getattr(pytorch_random, "randint", None)
    if original_randint is None or getattr(
        original_randint,
        "_pyrecest_integer_dtype_contract",
        False,
    ):
        return

    def _validate_randint_dtype(dtype):
        dtype = pytorch_random._normalize_random_dtype(dtype, default=None)
        if isinstance(dtype, torch.dtype) and dtype not in pytorch_random._INTEGER_DTYPES:
            raise TypeError("randint dtype must be an integer dtype")
        return dtype

    def randint(low, high=None, size=None, *args, **kwargs):
        if "dtype" in kwargs:
            kwargs = dict(kwargs)
            kwargs["dtype"] = _validate_randint_dtype(kwargs["dtype"])
        return original_randint(low, high, size, *args, **kwargs)

    randint.__name__ = getattr(original_randint, "__name__", "randint")
    randint.__doc__ = getattr(original_randint, "__doc__", None)
    randint._pyrecest_integer_dtype_contract = True
    pytorch_random.randint = randint

    try:
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - import fails before this module
        return
    if getattr(backend, "__backend_name__", None) == "pytorch":
        backend.random.randint = randint


def patch_pytorch_dtype_promotion_contract() -> None:
    """Make PyTorch mixed-dtype helpers use Torch's promotion rules."""
    try:
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend import failed earlier
        return

    _patch_pytorch_randint_dtype_contract(raw_pytorch, torch)

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
