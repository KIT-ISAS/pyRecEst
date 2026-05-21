#!/usr/bin/env python
"""Execute Python fenced code blocks from selected Markdown files."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

PYTHON_FENCE_RE = re.compile(r"```(?:python|py)\n(?P<code>.*?)\n```", re.DOTALL)
SKIP_MARKERS = ("# pyrecest: skip", "# doctest: +SKIP")


@dataclass(frozen=True)
class CodeBlock:
    path: Path
    index: int
    code: str


def iter_python_blocks(path: Path):
    text = path.read_text(encoding="utf-8")
    for index, match in enumerate(PYTHON_FENCE_RE.finditer(text), start=1):
        code = match.group("code").strip()
        if not code or any(marker in code for marker in SKIP_MARKERS):
            continue
        if "..." in code:
            continue
        yield CodeBlock(path=path, index=index, code=code)


def run_block(block: CodeBlock, *, env: dict[str, str]) -> int:
    with tempfile.NamedTemporaryFile("w", suffix=".py", encoding="utf-8", delete=False) as handle:
        handle.write(block.code)
        script_path = Path(handle.name)
    try:
        completed = subprocess.run([sys.executable, str(script_path)], env=env, text=True)
        return int(completed.returncode)
    finally:
        script_path.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--collect-only", action="store_true", help="List runnable blocks without executing them.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop at the first failing block.")
    args = parser.parse_args(argv)

    env = os.environ.copy()
    src_path = str(Path.cwd() / "src")
    env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")

    failures = 0
    for path in args.paths:
        for block in iter_python_blocks(path):
            label = f"{block.path}:{block.index}"
            if args.collect_only:
                print(label)
                continue
            print(f"Running {label}")
            if run_block(block, env=env) != 0:
                failures += 1
                if args.fail_fast:
                    return 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
