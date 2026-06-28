import pytest

jax = pytest.importorskip("jax")
from pyrecest._backend.jax import random  # noqa: E402


def test_zero_sized_choice_accepts_nonpositive_integer_population():
    random.seed(0)

    zero_sample = random.choice(0, size=(0,))
    negative_sample = random.choice(-1, size=(0,))

    assert zero_sample.shape == (0,)
    assert negative_sample.shape == (0,)
