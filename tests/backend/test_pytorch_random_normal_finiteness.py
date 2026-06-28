import math

import pytest

torch = pytest.importorskip("torch")

from pyrecest._backend.pytorch import random  # noqa: E402


@pytest.mark.parametrize(
    ("loc", "scale", "message"),
    [
        (math.inf, 1.0, "loc"),
        (math.nan, 1.0, "loc"),
        (torch.tensor([0.0, math.inf]), 1.0, "loc"),
        (0.0, math.inf, "scale"),
        (0.0, math.nan, "scale"),
        (0.0, torch.tensor([1.0, math.inf]), "scale"),
    ],
)
def test_normal_rejects_non_finite_parameters(loc, scale, message):
    with pytest.raises(ValueError, match=message):
        random.normal(loc, scale)
