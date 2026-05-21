#!/usr/bin/env python3
"""Validate benchmark output produced by benchmarks/basic_regressions.py."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("benchmark payload must be a JSON object")
    return payload


def validate_payload(
    payload: dict[str, Any], *, max_elapsed_seconds: float | None
) -> None:
    benchmarks = payload.get("benchmarks")
    if not isinstance(benchmarks, list) or not benchmarks:
        raise ValueError("benchmark payload must contain a non-empty 'benchmarks' list")

    for entry in benchmarks:
        if not isinstance(entry, dict):
            raise ValueError("each benchmark entry must be a JSON object")
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("each benchmark entry must have a non-empty string 'name'")
        elapsed = entry.get("elapsed_seconds")
        if not isinstance(elapsed, int | float) or elapsed < 0:
            raise ValueError(
                f"benchmark {name!r} must have a non-negative numeric elapsed_seconds"
            )
        if max_elapsed_seconds is not None and elapsed > max_elapsed_seconds:
            raise ValueError(
                f"benchmark {name!r} elapsed_seconds={elapsed:.6g} exceeds max_elapsed_seconds={max_elapsed_seconds:.6g}"
            )
        if "final_estimate" not in entry:
            raise ValueError(f"benchmark {name!r} must record a final_estimate")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Benchmark JSON file to validate.")
    parser.add_argument("--max-elapsed-seconds", type=float, default=None)
    args = parser.parse_args(argv)

    validate_payload(
        _load_payload(args.path), max_elapsed_seconds=args.max_elapsed_seconds
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
