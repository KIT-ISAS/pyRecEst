import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pyrecest._backend.pytorch import random  # noqa: E402


@pytest.mark.parametrize("control", ["replace", "shuffle"])
@pytest.mark.parametrize("value", [np.bool_(False), np.array(False)])
def test_choice_accepts_numpy_scalar_boolean_controls(control, value):
    kwargs = {control: value}
    size = 2
    if control == "shuffle":
        kwargs["replace"] = False
        size = 3

    samples = random.choice(torch.arange(3), size=size, **kwargs)

    assert samples.shape == (size,)


@pytest.mark.parametrize("control", ["replace", "shuffle"])
def test_choice_rejects_numpy_boolean_vector_controls(control):
    kwargs = {control: np.array([False])}

    with pytest.raises(TypeError, match=f"{control} must be a boolean"):
        random.choice(torch.arange(3), size=2, **kwargs)
