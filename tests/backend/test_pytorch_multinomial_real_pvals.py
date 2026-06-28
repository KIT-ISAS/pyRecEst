import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pyrecest._backend.pytorch import random  # noqa: E402


@pytest.mark.parametrize(
    "pvals",
    [
        [True, False],
        np.array([True, False]),
        torch.tensor([True, False]),
    ],
)
def test_multinomial_rejects_boolean_pvals(pvals):
    with pytest.raises(TypeError, match="real numeric"):
        random.multinomial(1, pvals)


@pytest.mark.parametrize(
    "pvals",
    [
        [1.0 + 0.0j],
        np.array([1.0 + 0.0j]),
        torch.tensor([1.0 + 0.0j]),
    ],
)
def test_multinomial_rejects_complex_pvals(pvals):
    with pytest.raises(TypeError, match="real numeric"):
        random.multinomial(1, pvals)


def test_multinomial_accepts_real_pvals():
    random.seed(0)

    result = random.multinomial(2, [1.0, 0.0])

    assert result.shape == (2,)
    assert int(result.sum()) == 2
