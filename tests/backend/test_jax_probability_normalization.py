import numpy as np
import pytest

pytest.importorskip("jax")
from pyrecest._backend.jax import random  # noqa: E402


def test_choice_accepts_tiny_probability_weights():
    random.seed(0)
    samples = random.choice([10, 20], size=8, p=[1e-300, 2e-300])
    assert samples.shape == (8,)
    assert set(np.asarray(samples).tolist()).issubset({10, 20})
