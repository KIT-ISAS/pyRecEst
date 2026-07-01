from __future__ import annotations

import importlib.util

import pytest


@pytest.mark.backend_portable
def test_jax_matrix_rank_accepts_hermitian_argument():
    if importlib.util.find_spec("jax") is None:
        pytest.skip("JAX is not installed")

    from pyrecest._backend.jax import linalg

    rank = linalg.matrix_rank([[1.0, 0.0], [0.0, 0.0]], hermitian=True)

    assert rank.item() == 1
