import numpy as np
import pytest

pytest.importorskip("autograd")

from pyrecest._backend.autograd import random  # noqa: E402


def test_multinomial_accepts_real_probability_values():
    draw = getattr(random, "multinomial")
    sample = draw(4, [0.25, 0.75])

    assert sample.shape == (2,)
    assert int(np.sum(sample)) == 4
