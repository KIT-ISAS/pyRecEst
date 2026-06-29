"""PyTorch dtype compatibility helpers."""

from __future__ import annotations

from pyrecest.backend_support._torch_sort_contract import patch_pytorch_sort_numpy_contract


def patch_pytorch_dtype_promotion_contract() -> None:
    """Make PyTorch mixed-dtype helpers use Torch-compatible promotion rules."""
    try:
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend import failed earlier
        return

    original_convert = raw_pytorch.convert_to_wider_dtype
    if not getattr(original_convert, "_pyrecest_torch_promotion_contract", False):

        def convert_to_wider_dtype(tensor_list):
            tensors = list(tensor_list)
            if not tensors:
                return tensors

            dtype = tensors[0].dtype
            for tensor in tensors[1:]:
                dtype = torch.result_type(
                    torch.empty((), dtype=dtype),
                    torch.empty((), dtype=tensor.dtype),
                )

            if all(tensor.dtype == dtype for tensor in tensors):
                return tensors
            return [raw_pytorch.cast(tensor, dtype=dtype) for tensor in tensors]

        convert_to_wider_dtype.__name__ = getattr(
            original_convert, "__name__", "convert_to_wider_dtype"
        )
        convert_to_wider_dtype.__doc__ = getattr(original_convert, "__doc__", None)
        convert_to_wider_dtype._pyrecest_torch_promotion_contract = True
        raw_pytorch.convert_to_wider_dtype = convert_to_wider_dtype

    patch_pytorch_sort_numpy_contract()
