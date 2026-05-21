#!/usr/bin/env python3
"""Smoke-test imports that should work from the default wheel installation."""

from __future__ import annotations

import argparse
import importlib

DEFAULT_IMPORTS = (
    "pyrecest",
    "pyrecest.backend",
    "pyrecest.distributions",
    "pyrecest.filters",
    "pyrecest.models",
    "pyrecest.sampling",
    "pyrecest.smoothers",
    "pyrecest.evaluation",
    "pyrecest.utils",
    "pyrecest.cli",
)


def check_imports(module_names: tuple[str, ...]) -> list[str]:
    failed: list[str] = []
    for module_name in module_names:
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - CLI smoke check
            failed.append(f"{module_name}: {exc}")
    return failed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("modules", nargs="*", help="Optional module names to import.")
    args = parser.parse_args(argv)
    modules = tuple(args.modules) if args.modules else DEFAULT_IMPORTS
    failed = check_imports(modules)
    if failed:
        for failure in failed:
            print(f"::error::{failure}")
        return 1
    print("Default-install import smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
