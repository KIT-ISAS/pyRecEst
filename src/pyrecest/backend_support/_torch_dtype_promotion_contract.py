from operator import index as _idx


def patch_pytorch_dtype_promotion_contract():
    try:
        import numpy as np
        import pyrecest._backend.pytorch as rp
        import pyrecest.backend as be
        import torch
    except ModuleNotFoundError:
        return
    _patch_repeat(rp, be, torch, np)
    _patch_diff(rp, be, torch)
    _patch_pad(rp, be, torch, np)
    _patch_cmp(rp, be, torch)
    _patch_dtype(rp, be, torch)


def _active(be):
    return getattr(be, "__backend_name__", None) == "pytorch"


def _pub(be, name, value):
    if _active(be):
        setattr(be, name, value)


def _patch_dtype(rp, be, torch):
    old = rp.convert_to_wider_dtype
    if getattr(old, "_pyrecest_torch_promotion_contract", False):
        _pub(be, "convert_to_wider_dtype", old)
        return

    def convert_to_wider_dtype(tensor_list):
        xs = list(tensor_list)
        if not xs:
            return xs
        dtype = xs[0].dtype
        for x in xs[1:]:
            dtype = torch.promote_types(dtype, x.dtype)
        if all(x.dtype == dtype for x in xs):
            return xs
        return [rp.cast(x, dtype=dtype) for x in xs]

    convert_to_wider_dtype.__name__ = getattr(old, "__name__", "convert_to_wider_dtype")
    convert_to_wider_dtype.__doc__ = getattr(old, "__doc__", None)
    convert_to_wider_dtype._pyrecest_torch_promotion_contract = True
    rp.convert_to_wider_dtype = convert_to_wider_dtype
    _pub(be, "convert_to_wider_dtype", convert_to_wider_dtype)


def _patch_cmp(rp, be, torch):
    funcs = {
        "greater": torch.greater,
        "less": torch.less,
        "less_equal": torch.less_equal,
        "logical_or": torch.logical_or,
    }
    if all(
        getattr(getattr(rp, name, None), "_pyrecest_arraylike_contract", False)
        for name in funcs
    ):
        for name in funcs:
            _pub(be, name, getattr(rp, name))
        return

    def coerce(x, y):
        dev = next((v.device for v in (x, y) if torch.is_tensor(v)), None)
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, device=dev)
        elif dev is not None and x.device != dev:
            x = x.to(device=dev)
        if not torch.is_tensor(y):
            y = torch.as_tensor(y, device=dev)
        elif dev is not None and y.device != dev:
            y = y.to(device=dev)
        return x, y

    def wrap(fn, name):
        def comparison(x, y, **kw):
            x, y = coerce(x, y)
            return fn(x, y, **kw)

        comparison.__name__ = name
        comparison.__doc__ = getattr(fn, "__doc__", None)
        comparison._pyrecest_arraylike_contract = True
        return comparison

    for name, fn in funcs.items():
        helper = wrap(fn, name)
        setattr(rp, name, helper)
        _pub(be, name, helper)


def _repeat_count(repetition):
    try:
        count = _idx(repetition)
    except TypeError as exc:
        raise TypeError("repeat counts must be integers") from exc
    if count < 0:
        raise ValueError("repeats may not contain negative values")
    return count


def _repeat_counts(repeats, *, np, torch, device):
    if torch.is_tensor(repeats):
        if repeats.ndim > 1:
            raise ValueError("object too deep for desired array")
        if repeats.dtype.is_floating_point or repeats.dtype.is_complex:
            raise TypeError("repeat counts must be integers")
        counts = repeats.to(device=device, dtype=torch.long)
        if bool(torch.any(counts < 0)):
            raise ValueError("repeats may not contain negative values")
        return counts
    arr = np.asarray(repeats)
    if arr.shape == ():
        return _repeat_count(arr.item())
    if arr.ndim > 1:
        raise ValueError("object too deep for desired array")
    if not np.can_cast(arr.dtype, np.dtype("intp"), casting="safe"):
        raise TypeError("repeat counts must be integers")
    counts = torch.as_tensor(arr, dtype=torch.long, device=device)
    if bool(torch.any(counts < 0)):
        raise ValueError("repeats may not contain negative values")
    return counts


def _patch_repeat(rp, be, torch, np):
    old = rp.repeat
    if getattr(old, "_pyrecest_numpy_contract", False):
        return

    def repeat(a, repeats, axis=None, *, dim=None, output_size=None):
        if dim is not None:
            if axis is not None and axis != dim:
                raise TypeError("repeat() got both 'axis' and 'dim'")
            axis = dim
        if axis is not None:
            axis = _idx(axis)
        values = rp.array(a)
        counts = _repeat_counts(repeats, np=np, torch=torch, device=values.device)
        kw = {"dim": axis}
        if output_size is not None:
            kw["output_size"] = output_size
        return old(values, counts, **kw)

    repeat.__name__ = getattr(old, "__name__", "repeat")
    repeat.__doc__ = getattr(old, "__doc__", None)
    repeat._pyrecest_numpy_contract = True
    rp.repeat = repeat
    _pub(be, "repeat", repeat)


def _patch_diff(rp, be, torch):
    old = rp.diff
    if getattr(old, "_pyrecest_numpy_contract", False):
        return
    none = object()

    def norm_axis(axis, ndim):
        axis = _idx(axis)
        if axis < 0:
            axis += ndim
        if axis < 0 or axis >= ndim:
            raise IndexError(f"axis {axis} is out of bounds for array of dimension {ndim}")
        return axis

    def boundary(value, ref, axis):
        out = rp.array(value)
        if out.device != ref.device:
            out = out.to(device=ref.device)
        if out.ndim == 0:
            shape = list(ref.shape)
            shape[axis] = 1
            out = torch.broadcast_to(out, tuple(shape))
        return out

    def diff(a, n=1, axis=-1, prepend=none, append=none):
        values = rp.array(a)
        order = _idx(n)
        if order < 0:
            raise ValueError(f"order must be non-negative but got {order}")
        if order == 0:
            return values.clone()
        if values.ndim == 0:
            raise ValueError("diff requires input that is at least one dimensional")
        axis = norm_axis(axis, values.ndim)
        parts = []
        if prepend is not none:
            parts.append(boundary(prepend, values, axis))
        parts.append(values)
        if append is not none:
            parts.append(boundary(append, values, axis))
        if len(parts) > 1:
            values = torch.cat(rp.convert_to_wider_dtype(parts), dim=axis)
        return torch.diff(values, n=order, dim=axis)

    diff.__name__ = getattr(old, "__name__", "diff")
    diff.__doc__ = getattr(old, "__doc__", None)
    diff._pyrecest_numpy_contract = True
    rp.diff = diff
    _pub(be, "diff", diff)


def _pad_pairs(pad_width, ndim, np):
    try:
        pairs = np.broadcast_to(np.asarray(pad_width), (ndim, 2))
    except ValueError as exc:
        raise ValueError(f"pad_width must be broadcastable to shape ({ndim}, 2)") from exc
    if np.any(pairs < 0):
        raise ValueError("index can't contain negative values")
    try:
        return tuple((_idx(before), _idx(after)) for before, after in pairs.tolist())
    except TypeError as exc:
        raise TypeError("pad_width must be of integral type") from exc


def _constant_pairs(constant_values, ndim, np):
    try:
        pairs = np.broadcast_to(np.asarray(constant_values), (ndim, 2))
    except ValueError as exc:
        raise ValueError(
            f"constant_values must be broadcastable to shape ({ndim}, 2)"
        ) from exc
    return tuple(tuple(pair) for pair in pairs.tolist())


def _block(shape, value, ref, torch):
    value = torch.as_tensor(value, dtype=ref.dtype, device=ref.device)
    if value.ndim != 0:
        raise ValueError("constant_values entries must be scalar")
    out = torch.empty(tuple(shape), dtype=ref.dtype, device=ref.device)
    out.fill_(value)
    return out


def _constant_pad(values, pad_width, constant_values, torch, np):
    pads = _pad_pairs(pad_width, values.ndim, np)
    constants = _constant_pairs(constant_values, values.ndim, np)
    out = values
    for axis, ((before, after), (before_value, after_value)) in enumerate(
        zip(pads, constants)
    ):
        if before:
            shape = list(out.shape)
            shape[axis] = before
            out = torch.cat((_block(shape, before_value, out, torch), out), dim=axis)
        if after:
            shape = list(out.shape)
            shape[axis] = after
            out = torch.cat((out, _block(shape, after_value, out, torch)), dim=axis)
    return out


def _patch_pad(rp, be, torch, np):
    old = rp.pad
    if getattr(old, "_pyrecest_constant_values_contract", False):
        return

    def pad(a, pad_width, mode="constant", constant_values=0.0):
        values = rp.array(a)
        if mode != "constant":
            return old(values, pad_width, mode=mode, constant_values=constant_values)
        return _constant_pad(values, pad_width, constant_values, torch, np)

    pad.__name__ = getattr(old, "__name__", "pad")
    pad.__doc__ = getattr(old, "__doc__", None)
    pad._pyrecest_constant_values_contract = True
    rp.pad = pad
    _pub(be, "pad", pad)
