import numpy as _np
from torch import (
    bool as _torch_bool,
    complex64 as _torch_complex64,
    complex128 as _torch_complex128,
    dtype as _torch_dtype,
    float16 as _torch_float16,
    float32 as _torch_float32,
    float64 as _torch_float64,
    from_numpy as _torch_from_numpy,
    int8 as _torch_int8,
    int16 as _torch_int16,
    int32 as _torch_int32,
    int64 as _torch_int64,
    is_tensor as _torch_is_tensor,
    stack as _torch_stack,
    tensor as _torch_tensor,
    uint8 as _torch_uint8,
)

_TORCH_DTYPE_BY_NAME = {
    "bool": _torch_bool,
    "uint8": _torch_uint8,
    "int8": _torch_int8,
    "int16": _torch_int16,
    "int32": _torch_int32,
    "int64": _torch_int64,
    "float16": _torch_float16,
    "float32": _torch_float32,
    "float64": _torch_float64,
    "complex64": _torch_complex64,
    "complex128": _torch_complex128,
}


def _normalize_dtype(dtype):
    """Return a torch dtype for dtype-like values understood by PyTorch/NumPy."""
    if dtype is None or isinstance(dtype, _torch_dtype):
        return dtype
    if isinstance(dtype, str) and dtype.startswith("torch."):
        dtype = dtype.split(".", 1)[1]
    try:
        return _TORCH_DTYPE_BY_NAME[str(_np.dtype(dtype))]
    except (KeyError, TypeError):
        return dtype


def from_numpy(x):
    if _torch_is_tensor(x):
        return x
    if isinstance(x, _np.ndarray) and any(stride < 0 for stride in x.strides):
        x = x.copy()
    return _torch_from_numpy(x)


def array(val, dtype=None):
    dtype = _normalize_dtype(dtype)
    if _torch_is_tensor(val):
        if dtype is None or val.dtype == dtype:
            return val.clone()

        return cast(val, dtype=dtype)

    if isinstance(val, _np.ndarray):
        tensor = from_numpy(val)
        if dtype is not None and tensor.dtype != dtype:
            tensor = cast(tensor, dtype=dtype)

        return tensor

    if isinstance(val, (list, tuple)) and len(val):
        tensors = [array(tensor, dtype=dtype) for tensor in val]
        return _torch_stack(tensors)

    return _torch_tensor(val, dtype=dtype)


def cast(x, dtype):
    dtype = _normalize_dtype(dtype)
    if _torch_is_tensor(x):
        return x.to(dtype=dtype)
    return array(x, dtype=dtype)
