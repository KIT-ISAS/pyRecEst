"""Regression test for JAX random integer argument handling."""

from __future__ import annotations

import importlib.util

import pytest

from tests.support.backend_runner import run_backend_code


_CHECK = """
import pyrecest.backend as backend
from pyrecest.backend import random

samples = random.randint(0, 5, size=(16,))
high_only = random.randint(5, size=(16,))
compat = random.randint((16,), minval=2, maxval=7)

assert samples.shape == (16,)
assert high_only.shape == (16,)
assert compat.shape == (16,)
assert bool(backend.all(samples >= 0))
assert bool(backend.all(samples < 5))
assert bool(backend.all(high_only >= 0))
assert bool(backend.all(high_only < 5))
assert bool(backend.all(compat >= 2))
assert bool(backend.all(compat < 7))
print("ok")
"""


@pytest.mark.backend_portable
def test_jax_randint_argument_styles():
    if importlib.util.find_spec("jax") is None:
        pytest.skip("JAX is not installed")

    result = run_backend_code("jax", _CHECK)

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
