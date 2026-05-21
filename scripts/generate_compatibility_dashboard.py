#!/usr/bin/env python
"""Generate a compact compatibility dashboard in Markdown."""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path

from pyrecest.backend_support import format_backend_support_markdown


def _python_range(pyproject_path: Path) -> str:
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    return data["tool"]["poetry"]["dependencies"]["python"]


def _scenario_names(root: Path) -> list[str]:
    scenarios = root / "scenarios"
    if not scenarios.exists():
        return []
    return sorted(path.name for path in scenarios.iterdir() if path.is_dir())


def render_dashboard(root: Path) -> str:
    lines = [
        "# Compatibility Dashboard",
        "",
        f"Python support declared in `pyproject.toml`: `{_python_range(root / 'pyproject.toml')}`.",
        "",
        "## Public Backend API Matrix",
        "",
        format_backend_support_markdown(),
        "",
        "## Scenario Zoo",
        "",
    ]
    scenarios = _scenario_names(root)
    if scenarios:
        lines.extend(f"- `{name}`" for name in scenarios)
    else:
        lines.append("No scenarios found.")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output", type=Path, default=Path("docs/compatibility-dashboard.md")
    )
    args = parser.parse_args(argv)

    text = render_dashboard(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
