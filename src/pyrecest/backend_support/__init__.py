"""Public accessors for backend support metadata."""

from __future__ import annotations

from pyrecest._backend.capabilities import (
    API_BACKEND_CAPABILITIES,
    BACKEND_SUPPORT_LEVELS,
    iter_api_backend_capabilities,
)


def _patch_pytorch_dot_numpy_contract() -> None:
    """Make PyTorch dot follow NumPy's contraction axes."""
    try:
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - import fails before this module
        return

    if getattr(backend, "__backend_name__", None) != "pytorch":
        return

    try:
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend import failed earlier
        return

    original_dot = raw_pytorch.dot
    if getattr(original_dot, "_pyrecest_numpy_contract", False):
        return

    def dot(a, b):
        a = backend.array(a)
        b = backend.array(b)
        dtype = torch.promote_types(a.dtype, b.dtype)
        a = a.to(dtype=dtype)
        b = b.to(dtype=dtype)

        if a.ndim == 0 or b.ndim == 0:
            return torch.multiply(a, b)
        if a.ndim == 1 and b.ndim == 1:
            return torch.dot(a, b)
        if b.ndim == 1:
            return torch.tensordot(a, b, dims=([-1], [0]))
        if a.ndim == 1:
            return torch.tensordot(a, b, dims=([0], [-2]))
        return torch.tensordot(a, b, dims=([-1], [-2]))

    dot.__name__ = getattr(original_dot, "__name__", "dot")
    dot.__doc__ = getattr(original_dot, "__doc__", None)
    dot._pyrecest_numpy_contract = True
    backend.dot = dot
    raw_pytorch.dot = dot


_patch_pytorch_dot_numpy_contract()


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
