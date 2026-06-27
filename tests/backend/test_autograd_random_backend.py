import numpy as np
import pytest

pytest.importorskip("autograd")

from pyrecest._backend.autograd import random  # noqa: E402


def test_multinomial_accepts_real_probability_values():
    draw = getattr(random, "multinomial")
    sample = draw(4, [0.25, 0.75])

    assert sample.shape == (2,)
    assert int(np.sum(sample)) == 4


def test_multinomial_rejects_non_numeric_dtype_probability_values():
    draw = getattr(random, "multinomial")
    pvals = np.array([1, 0], dtype="?")

    with pytest.raises(TypeError, match="real numeric"):
        draw(4, pvals)
