#!/usr/bin/env python
"""Render the backend API capability matrix as Markdown.

Run from a source checkout with ``PYTHONPATH=src`` or inside the Poetry
environment.  The script intentionally reads from the same source of truth as
the CLI and tests: ``pyrecest._backend.capabilities``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pyrecest._backend.capabilities import API_BACKEND_CAPABILITIES


def render_markdown() -> str:
    lines = [
        "| API | NumPy | PyTorch | JAX | Notes |",
        "|-----|-------|---------|-----|-------|",
    ]
    for api_name, row in sorted(API_BACKEND_CAPABILITIES.items()):
        lines.append(
            "| `{api}` | {numpy} | {pytorch} | {jax} | {notes} |".format(
                api=api_name,
                numpy=row.get("numpy", "unknown"),
                pytorch=row.get("pytorch", "unknown"),
                jax=row.get("jax", "unknown"),
                notes=row.get("notes", ""),
            )
        )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output path. If omitted, write to stdout.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    markdown = render_markdown()
    if args.output:
        args.output.write_text(markdown, encoding="utf-8")
    else:
        print(markdown, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
