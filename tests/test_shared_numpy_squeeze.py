import pyrecest.backend as backend
import pytest
from pyrecest.backend import array


def _to_python(value):
    value = backend.to_numpy(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def test_shared_numpy_squeeze_accepts_tuple_axis():
    if backend.__backend_name__ not in ("numpy", "autograd"):
        pytest.skip("shared NumPy/autograd squeeze regression test")

    result = backend.squeeze(array([[[1], [2]]]), axis=(0, 2))

    assert result.shape == (2,)
    assert _to_python(result) == [1, 2]


@pytest.mark.parametrize("axis", [3, -4])
def test_shared_numpy_squeeze_rejects_out_of_bounds_axis(axis):
    if backend.__backend_name__ not in ("numpy", "autograd"):
        pytest.skip("shared NumPy/autograd squeeze regression test")

    with pytest.raises(Exception) as exc_info:
        backend.squeeze(array([[[1], [2]]]), axis=axis)

    assert "out of bounds" in str(exc_info.value)
