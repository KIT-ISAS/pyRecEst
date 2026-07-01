import numpy as np
import pytest

try:
    from pyrecest._backend import pytorch as pytorch_backend
except ModuleNotFoundError:
    pytorch_backend = None


@pytest.mark.skipif(pytorch_backend is None, reason="PyTorch is not installed")
@pytest.mark.parametrize(
    "axis",
    (np.array(0), np.array([0])),
    ids=("zero_dim_array", "single_element_array"),
)
def test_linalg_norm_accepts_numpy_scalar_array_axis(axis):
    values = pytorch_backend.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
    expected = pytorch_backend.array([3.0, np.sqrt(17.0), np.sqrt(29.0)])

    result = pytorch_backend.linalg.norm(values, axis=axis)

    assert pytorch_backend.allclose(result, expected)
