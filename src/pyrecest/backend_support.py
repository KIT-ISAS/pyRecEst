"""Public accessors for backend support metadata."""

from __future__ import annotations

from pyrecest._backend.capabilities import (
    API_BACKEND_CAPABILITIES,
    BACKEND_SUPPORT_LEVELS,
    iter_api_backend_capabilities,
)


def get_backend_support(
    api_name: str, *, backend: str | None = None
) -> dict[str, str] | str | None:
    """Return backend support metadata for a public API.

    Parameters
    ----------
    api_name:
        Name as listed in the backend API matrix.
    backend:
        Optional backend name. When supplied, return only that backend's support
        level. Otherwise return the complete row.
    """
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
    """Return ``value`` escaped for use inside a Markdown table cell."""
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def format_backend_support_markdown() -> str:
    """Render the public backend API matrix as a Markdown table."""
    lines = [
        "| API | NumPy | PyTorch | JAX | Notes |",
        "|-----|-------|---------|-----|-------|",
    ]
    for api_name, row in iter_api_backend_capabilities():
        cells = [
            f"`{_markdown_table_cell(api_name)}`",
            _markdown_table_cell(row["numpy"]),
            _markdown_table_cell(row["pytorch"]),
            _markdown_table_cell(row["jax"]),
            _markdown_table_cell(row.get("notes", "")),
        ]
        lines.append(f"| {' | '.join(cells)} |")
    return "\n".join(lines)


__all__ = [
    "BACKEND_SUPPORT_LEVELS",
    "backend_support",
    "format_backend_support_markdown",
    "get_backend_support",
]
