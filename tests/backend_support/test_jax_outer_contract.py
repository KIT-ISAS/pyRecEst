"""Regression coverage for JAX outer PyRecEst semantics."""

from __future__ import annotations

import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code


@pytest.mark.backend_portable
def test_jax_outer_pairs_leading_dimensions():
    if importlib.util.find_spec("jax") is None:
        pytest.skip("JAX is not installed")

    result = run_backend_code(
        "jax",
        """
import pyrecest  # noqa: F401
import pyrecest.backend as backend
import pyrecest._backend.jax as raw_jax

expected = [
    [[10, 20], [20, 40]],
    [[90, 120], [120, 160]],
]

public_result = backend.outer([[1, 2], [3, 4]], [[10, 20], [30, 40]])
assert tuple(public_result.shape) == (2, 2, 2)
assert backend.to_numpy(public_result).tolist() == expected

raw_result = raw_jax.outer([[1, 2], [3, 4]], [[10, 20], [30, 40]])
assert tuple(raw_result.shape) == (2, 2, 2)
assert raw_jax.to_numpy(raw_result).tolist() == expected
""",
    )

    assert result.returncode == 0, result.stderr
