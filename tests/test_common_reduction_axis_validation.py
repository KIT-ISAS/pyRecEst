import importlib.util

import numpy as np
import pytest

from pyrecest._backend import _common


@pytest.mark.backend_portable
def test_common_torch_min_rejects_boolean_axes():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    import torch

    values = torch.arange(6).reshape(2, 3)
    for axis in (True, False, (True,), [False]):
        with pytest.raises(TypeError):
            _common.min(values, axis=axis)


@pytest.mark.backend_portable
def test_common_torch_min_accepts_numpy_scalar_integer_axis():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    import torch

    values = torch.arange(6).reshape(2, 3)
    assert _common.min(values, axis=np.array(0)).tolist() == [0, 1, 2]
