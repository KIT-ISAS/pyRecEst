"""Public accessors for backend support metadata."""

from __future__ import annotations

from operator import index as _operator_index

from pyrecest._backend.capabilities import (
    API_BACKEND_CAPABILITIES,
    BACKEND_SUPPORT_LEVELS,
    iter_api_backend_capabilities,
)
from pyrecest.backend_support._torch_dtype_promotion_contract import (
    patch_pytorch_dtype_promotion_contract as _patch_pytorch_dtype_promotion_contract,
)


def _pytorch_scalar_tensor_index(index, torch_module):
    """Return Python int indices for scalar integer tensors."""

    if not torch_module.is_tensor(index) or index.ndim != 0:
        return index
    if (
        index.dtype in {torch_module.bool, torch_module.uint8}
        or index.dtype.is_floating_point
        or index.dtype.is_complex
    ):
        return index
    return _operator_index(index)


def _wrap_pytorch_assignment_helper(original_assignment, torch_module):
    """Normalize scalar tensor indices before assignment helper len() checks."""

    if getattr(original_assignment, "_pyrecest_scalar_tensor_index_contract", False):
        return original_assignment

    def assignment(x, values, indices, axis=0):
        indices = _pytorch_scalar_tensor_index(indices, torch_module)
        return original_assignment(x, values, indices, axis=axis)

    assignment.__name__ = getattr(original_assignment, "__name__", "assignment")
    assignment.__doc__ = getattr(original_assignment, "__doc__", None)
    assignment._pyrecest_scalar_tensor_index_contract = True
    return assignment


def _patch_raw_pytorch_assignment_scalar_tensor_indices() -> None:
    """Make raw PyTorch assignment helpers accept scalar integer tensor indices."""

    try:
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import torch as _torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend may be unavailable
        return

    raw_pytorch.assignment = _wrap_pytorch_assignment_helper(
        raw_pytorch.assignment,
        _torch,
    )
    raw_pytorch.assignment_by_sum = _wrap_pytorch_assignment_helper(
        raw_pytorch.assignment_by_sum,
        _torch,
    )
    if getattr(backend, "__backend_name__", None) == "pytorch":
        backend.assignment = raw_pytorch.assignment
        backend.assignment_by_sum = raw_pytorch.assignment_by_sum


def _patch_pytorch_dot_numpy_contract() -> None:
    """Make PyTorch dot follow the backend batched inner-product contract."""
    try:
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - import fails before this module
        return

    active_pytorch_backend = getattr(backend, "__backend_name__", None) == "pytorch"

    try:
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
    except (
        ModuleNotFoundError
    ):  # pragma: no cover - PyTorch backend import failed earlier
        return

    original_dot = raw_pytorch.dot
    if getattr(original_dot, "_pyrecest_numpy_contract", False):
        return

    def dot(a, b):
        a = raw_pytorch.array(a)
        b = raw_pytorch.array(b)
        dtype = torch.promote_types(a.dtype, b.dtype)
        a = a.to(dtype=dtype)
        b = b.to(dtype=dtype)

        if a.ndim == 0 or b.ndim == 0:
            return torch.multiply(a, b)
        if a.ndim == 1 and b.ndim == 1:
            return torch.dot(a, b)
        if b.ndim == 1:
            return torch.einsum("...i,i->...", a, b)
        if a.ndim == 1:
            return torch.einsum("i,...i->...", a, b)
        return torch.einsum("...i,...i->...", a, b)

    dot.__name__ = getattr(original_dot, "__name__", "dot")
    dot.__doc__ = getattr(original_dot, "__doc__", None)
    dot._pyrecest_numpy_contract = True
    raw_pytorch.dot = dot
    if active_pytorch_backend:
        backend.dot = dot


def _patch_pytorch_outer_numpy_contract() -> None:
    """Make PyTorch outer pair leading dimensions like the backend contract."""
    try:
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - import fails before this module
        return

    active_pytorch_backend = getattr(backend, "__backend_name__", None) == "pytorch"

    try:
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
    except (
        ModuleNotFoundError
    ):  # pragma: no cover - PyTorch backend import failed earlier
        return

    original_outer = raw_pytorch.outer
    if getattr(original_outer, "_pyrecest_numpy_contract", False):
        return

    def outer(a, b):
        a = raw_pytorch.array(a)
        b = raw_pytorch.array(b)
        dtype = torch.promote_types(a.dtype, b.dtype)
        a = a.to(dtype=dtype)
        b = b.to(dtype=dtype)
        if a.ndim == 0 or b.ndim == 0:
            return torch.multiply(a, b)
        return a[..., :, None] * b[..., None, :]

    outer.__name__ = getattr(original_outer, "__name__", "outer")
    outer.__doc__ = getattr(original_outer, "__doc__", None)
    outer._pyrecest_numpy_contract = True
    raw_pytorch.outer = outer
    if active_pytorch_backend:
        backend.outer = outer


def _pytorch_tile_repetition(repetition) -> int:
    """Return one NumPy-style tile repetition as an integer."""

    try:
        return _operator_index(repetition)
    except TypeError as exc:
        raise TypeError("tile repetitions must be integers") from exc


def _pytorch_tile_repetitions(reps, numpy_module, torch_module) -> tuple[int, ...]:
    """Normalize NumPy-style tile repetitions for ``torch.Tensor.repeat``."""

    if torch_module.is_tensor(reps):
        reps = reps.detach().cpu().numpy()
    reps_array = numpy_module.asarray(reps)
    if reps_array.shape == ():
        repetitions = (_pytorch_tile_repetition(reps_array.item()),)
    else:
        repetitions = tuple(
            _pytorch_tile_repetition(one_repetition)
            for one_repetition in reps_array.tolist()
        )
    if any(one_repetition < 0 for one_repetition in repetitions):
        raise ValueError("negative dimensions are not allowed")
    return repetitions


def _patch_pytorch_tile_numpy_contract() -> None:
    """Make PyTorch tile follow NumPy repetition semantics."""
    try:
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - import fails before this module
        return

    active_pytorch_backend = getattr(backend, "__backend_name__", None) == "pytorch"

    try:
        import numpy as np  # pylint: disable=import-outside-toplevel
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
    except (
        ModuleNotFoundError
    ):  # pragma: no cover - PyTorch backend import failed earlier
        return

    original_tile = raw_pytorch.tile
    if getattr(original_tile, "_pyrecest_numpy_contract", False):
        return

    def tile(x, reps):
        x = raw_pytorch.array(x)
        repetitions = _pytorch_tile_repetitions(reps, np, torch)
        if not repetitions:
            return x.clone()
        if x.ndim < len(repetitions):
            x = x.reshape((1,) * (len(repetitions) - x.ndim) + tuple(x.shape))
        elif x.ndim > len(repetitions):
            repetitions = (1,) * (x.ndim - len(repetitions)) + repetitions
        return x.repeat(repetitions)

    tile.__name__ = getattr(original_tile, "__name__", "tile")
    tile.__doc__ = getattr(np.tile, "__doc__", None)
    tile._pyrecest_numpy_contract = True
    raw_pytorch.tile = tile
    if active_pytorch_backend:
        backend.tile = tile


def _patch_pytorch_copy_numpy_contract() -> None:
    """Make PyTorch copy return tensors for array-like inputs."""
    try:
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - import fails before this module
        return

    active_pytorch_backend = getattr(backend, "__backend_name__", None) == "pytorch"

    try:
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend import failed earlier
        return

    original_copy = raw_pytorch.copy
    if getattr(original_copy, "_pyrecest_numpy_contract", False):
        return

    def copy(x):
        if raw_pytorch.is_array(x):
            return original_copy(x)
        return raw_pytorch.array(x)

    copy.__name__ = getattr(original_copy, "__name__", "copy")
    copy.__doc__ = getattr(original_copy, "__doc__", None)
    copy._pyrecest_numpy_contract = True
    raw_pytorch.copy = copy
    if active_pytorch_backend:
        backend.copy = copy


def _patch_pytorch_clip_numpy_contract() -> None:
    """Make PyTorch clip accept array-like inputs regardless of public backend."""
    try:
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - import fails before this module
        return

    active_pytorch_backend = getattr(backend, "__backend_name__", None) == "pytorch"

    try:
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend import failed earlier
        return

    original_clip = raw_pytorch.clip
    if getattr(original_clip, "_pyrecest_numpy_contract", False):
        return

    def _clip_bound(value, *, device):
        if value is None:
            return None
        if torch.is_tensor(value):
            return value.to(device=device)
        return torch.as_tensor(value, device=device)

    def clip(a, a_min=None, a_max=None, out=None, *, min=None, max=None):
        if min is not None:
            if a_min is not None:
                raise TypeError("clip() got both 'a_min' and 'min'")
            a_min = min
        if max is not None:
            if a_max is not None:
                raise TypeError("clip() got both 'a_max' and 'max'")
            a_max = max
        if a_min is None and a_max is None:
            raise ValueError("One of max or min must be given")

        x = raw_pytorch.array(a)
        result = torch.clip(
            x,
            min=_clip_bound(a_min, device=x.device),
            max=_clip_bound(a_max, device=x.device),
        )
        if out is not None:
            copy_ = getattr(out, "copy_", None)
            if copy_ is not None:
                copy_(result)
                return out
            out[...] = raw_pytorch.to_numpy(result)
            return out
        return result

    clip.__name__ = getattr(original_clip, "__name__", "clip")
    clip.__doc__ = getattr(original_clip, "__doc__", None)
    clip._pyrecest_numpy_contract = True
    raw_pytorch.clip = clip
    if active_pytorch_backend:
        backend.clip = clip


def _patch_pytorch_cov_numpy_contract() -> None:
    """Make PyTorch cov accept NumPy-style y/rowvar/ddof/dtype arguments."""
    try:
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - import fails before this module
        return

    active_pytorch_backend = getattr(backend, "__backend_name__", None) == "pytorch"

    try:
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend import failed earlier
        return

    original_cov = raw_pytorch.cov
    if getattr(original_cov, "_pyrecest_numpy_contract", False):
        return

    def _cov_device(*values):
        tensor_values = [value for value in values if torch.is_tensor(value)]
        if not tensor_values:
            return None
        return next(
            (value.device for value in tensor_values if value.device.type != "cpu"),
            tensor_values[0].device,
        )

    def _cov_tensor(values, *, dtype=None, device=None, name="m"):
        tensor = raw_pytorch.array(values, dtype=dtype)
        if device is not None and tensor.device != device:
            tensor = tensor.to(device=device)
        if tensor.ndim > 2:
            raise ValueError(f"{name} has more than 2 dimensions")
        return tensor

    def _cov_dtype(tensors, explicit_dtype=None):
        if explicit_dtype is not None:
            return tensors[0].dtype
        result_dtype = tensors[0].dtype
        for tensor in tensors[1:]:
            result_dtype = torch.promote_types(result_dtype, tensor.dtype)
        if result_dtype.is_floating_point or result_dtype.is_complex:
            return result_dtype
        return raw_pytorch.get_default_dtype()

    def _cov_matrix(tensor, *, rowvar):
        if tensor.ndim == 0:
            tensor = tensor.reshape(1)
        if tensor.ndim == 1:
            return tensor.reshape(1, -1)
        if not rowvar:
            return tensor.T
        return tensor

    def _cov_weights(weights, *, dtype=None, device=None):
        if weights is None:
            return None
        if torch.is_tensor(weights):
            result = weights.to(device=device)
            if dtype is not None:
                result = result.to(dtype=dtype)
            return result
        return torch.as_tensor(weights, dtype=dtype, device=device)

    def cov(
        m,
        y=None,
        rowvar=True,
        bias=False,
        ddof=None,
        fweights=None,
        aweights=None,
        dtype=None,
        *,
        correction=None,
    ):
        if (
            y is not None
            and correction is None
            and rowvar is True
            and ddof is None
            and fweights is None
            and aweights is None
            and dtype is None
            and not torch.is_tensor(y)
        ):
            try:
                correction = _operator_index(y)
            except TypeError:
                pass
            else:
                y = None

        device = _cov_device(m, y, fweights, aweights)
        m_tensor = _cov_tensor(m, dtype=dtype, device=device, name="m")
        cov_tensors = [m_tensor]
        y_tensor = None
        if y is not None:
            y_tensor = _cov_tensor(y, dtype=dtype, device=device, name="y")
            cov_tensors.append(y_tensor)

        result_dtype = _cov_dtype(cov_tensors, explicit_dtype=dtype)
        input_tensor = _cov_matrix(m_tensor.to(dtype=result_dtype), rowvar=rowvar)
        if y_tensor is not None:
            y_tensor = _cov_matrix(y_tensor.to(dtype=result_dtype), rowvar=rowvar)
            if y_tensor.shape[1] != input_tensor.shape[1]:
                raise ValueError("m and y have incompatible numbers of observations")
            input_tensor = torch.cat((input_tensor, y_tensor), dim=0)

        if ddof is not None and correction is not None:
            raise ValueError("ddof and correction cannot both be specified")
        if ddof is not None:
            correction_value = ddof
        elif bias:
            correction_value = 0
        elif correction is not None:
            correction_value = correction
        else:
            correction_value = 1

        fweights_tensor = _cov_weights(fweights, device=input_tensor.device)
        aweights_tensor = _cov_weights(
            aweights, dtype=input_tensor.dtype, device=input_tensor.device
        )
        return torch.cov(
            input_tensor,
            correction=_operator_index(correction_value),
            fweights=fweights_tensor,
            aweights=aweights_tensor,
        )

    cov.__name__ = getattr(original_cov, "__name__", "cov")
    cov.__doc__ = getattr(original_cov, "__doc__", None)
    cov._pyrecest_numpy_contract = True
    raw_pytorch.cov = cov
    if active_pytorch_backend:
        backend.cov = cov


def _patch_jax_outer_numpy_contract() -> None:
    """Make JAX outer pair leading dimensions like the backend contract."""
    try:
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - import fails before this module
        return

    active_jax_backend = getattr(backend, "__backend_name__", None) == "jax"

    try:
        import jax.numpy as jnp  # pylint: disable=import-outside-toplevel
        import pyrecest._backend.jax as raw_jax  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - JAX backend import failed earlier
        return

    original_outer = raw_jax.outer
    if getattr(original_outer, "_pyrecest_numpy_contract", False):
        return

    def outer(a, b):
        a = jnp.asarray(a)
        b = jnp.asarray(b)
        if a.ndim == 0 or b.ndim == 0:
            return jnp.multiply(a, b)
        return a[..., :, None] * b[..., None, :]

    outer.__name__ = getattr(original_outer, "__name__", "outer")
    outer.__doc__ = getattr(jnp.outer, "__doc__", None)
    outer._pyrecest_numpy_contract = True
    raw_jax.outer = outer
    if active_jax_backend:
        backend.outer = outer


_patch_raw_pytorch_assignment_scalar_tensor_indices()
_patch_pytorch_dtype_promotion_contract()
_patch_pytorch_dot_numpy_contract()
_patch_pytorch_outer_numpy_contract()
_patch_pytorch_tile_numpy_contract()
_patch_pytorch_copy_numpy_contract()
_patch_pytorch_clip_numpy_contract()
_patch_pytorch_cov_numpy_contract()
_patch_jax_outer_numpy_contract()


def get_backend_support(
    api_name: str, *, backend: str | None = None
) -> dict[str, str] | str | None:
    """Return backend support metadata for a public API."""
    row = API_BACKEND_CAPABILITIES.get(api_name)
    if row is None:
        return None
    if backend is not None:
        return row.get(backend)
    return dict(row)


def backend_support(
    api_name: str, backend: str | None = None
) -> dict[str, str] | str | None:
    """Alias for :func:`get_backend_support` for concise user code."""
    return get_backend_support(api_name, backend=backend)


def _markdown_table_cell(value: object) -> str:
    escape = chr(92) + chr(124)
    return str(value).replace("\r", " ").replace("\n", "<br>").replace(chr(124), escape)


def _markdown_table_row(cells: list[str]) -> str:
    separator = chr(124)
    return f"{separator} " + f" {separator} ".join(cells) + f" {separator}"


def format_backend_support_markdown() -> str:
    """Render the public backend API matrix as a Markdown table."""
    lines = [
        _markdown_table_row(["API", "NumPy", "PyTorch", "JAX", "Notes"]),
        _markdown_table_row(["-----", "-------", "---------", "-----", "-------"]),
    ]
    for api_name, row in iter_api_backend_capabilities():
        lines.append(
            _markdown_table_row(
                [
                    f"`{_markdown_table_cell(api_name)}`",
                    _markdown_table_cell(row["numpy"]),
                    _markdown_table_cell(row["pytorch"]),
                    _markdown_table_cell(row["jax"]),
                    _markdown_table_cell(row.get("notes", "")),
                ]
            )
        )
    return "\n".join(lines)


__all__ = [
    "BACKEND_SUPPORT_LEVELS",
    "backend_support",
    "format_backend_support_markdown",
    "get_backend_support",
]
