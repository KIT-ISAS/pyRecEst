import pytest

torch = pytest.importorskip("torch")

from pyrecest._backend.pytorch import random  # noqa: E402


@pytest.mark.parametrize("scale", [-1.0, -1])
def test_normal_rejects_negative_scalar_scale(scale):
    with pytest.raises(ValueError, match="scale"):
        random.normal(0.0, scale)
