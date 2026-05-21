#!/usr/bin/env python
"""Generate lightweight grouped release notes from git history."""

from __future__ import annotations

import argparse
import subprocess
from collections import defaultdict

GROUPS = {
    "feat": "Features",
    "fix": "Fixes",
    "docs": "Documentation",
    "test": "Tests",
    "perf": "Performance",
    "build": "Build and packaging",
    "ci": "Continuous integration",
    "refactor": "Refactoring",
    "chore": "Maintenance",
}


def _git_log(revision_range: str) -> list[str]:
    completed = subprocess.run(
        ["git", "log", "--format=%s", revision_range],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def _group_subject(subject: str) -> str:
    prefix = subject.split(":", 1)[0].split("(", 1)[0].lower()
    return GROUPS.get(prefix, "Other changes")


def render_release_notes(subjects: list[str]) -> str:
    grouped: dict[str, list[str]] = defaultdict(list)
    for subject in subjects:
        grouped[_group_subject(subject)].append(subject)

    lines = ["# Release Notes", ""]
    for group in [*GROUPS.values(), "Other changes"]:
        entries = grouped.get(group, [])
        if not entries:
            continue
        lines.extend([f"## {group}", ""])
        lines.extend(f"- {entry}" for entry in entries)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "revision_range", help="Git revision range, for example v2.2.1..HEAD"
    )
    parser.add_argument("--output", help="Optional output Markdown path")
    args = parser.parse_args(argv)

    rendered = render_release_notes(_git_log(args.revision_range))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(rendered)
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
