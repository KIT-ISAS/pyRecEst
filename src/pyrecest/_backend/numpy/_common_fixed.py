import functools

import numpy as _np
from pyrecest._backend import _backend_config as _config
from pyrecest._backend._dtype_utils import (
    _modify_func_default_dtype,
    get_default_cdtype,
    get_default_dtype,
)

from .._shared_numpy._common import (
    _add_default_dtype_by_casting,
    _allow_complex_dtype,
    _box_binary_scalar,
    _box_unary_scalar,
    _cast_fout_to_input_dtype,
    _cast_out_from_dtype,
    _cast_out_to_input_dtype,
    _get_wider_dtype,
    _is_boolean,
    _is_iterable,
    as_dtype,
    atol,
    cast,
    convert_to_wider_dtype,
    is_array,
    is_bool,
    is_complex,
    is_floating,
    rtol,
    set_default_dtype,
    to_ndarray,
)


def _dyn_update_dtype(dtype_pos=None, target=None):
    """Supply the NumPy default dtype only when the caller omitted it."""

    def _decorator(func):
        @functools.wraps(func)
        def _wrapped(*args, **kwargs):
            if dtype_pos is not None and len(args) > dtype_pos:
                if args[dtype_pos] is None:
                    args = list(args)
                    args[dtype_pos] = _config.DEFAULT_DTYPE
            elif kwargs.get("dtype") is None:
                kwargs["dtype"] = _config.DEFAULT_DTYPE

            return func(*args, **kwargs)

        return _wrapped

    if target is None:
        return _decorator

    return _decorator(target)


array = _cast_out_from_dtype(target=_np.array, dtype_pos=1)
eye = _modify_func_default_dtype(target=_np.eye)
zeros = _dyn_update_dtype(target=_np.zeros, dtype_pos=1)
