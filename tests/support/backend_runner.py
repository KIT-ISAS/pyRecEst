"""Subprocess helpers for import-time backend portability tests."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class BackendRunResult:
    backend: str
    returncode: int
    stdout: str
    stderr: str


def run_backend_code(backend: str, code: str, *, timeout: float = 30.0) -> BackendRunResult:
    env = os.environ.copy()
    env["PYRECEST_BACKEND"] = backend
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
    )
    return BackendRunResult(
        backend=backend,
        returncode=int(completed.returncode),
        stdout=completed.stdout,
        stderr=completed.stderr,
    )
