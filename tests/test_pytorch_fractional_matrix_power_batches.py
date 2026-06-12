import pyrecest.backend as backend
import pytest
from pyrecest.backend import linalg


def _to_python(value):
    value = backend.to_numpy(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def test_pytorch_fractional_matrix_power_accepts_multiple_batch_axes():
    if backend.__backend_name__ != "pytorch":
        pytest.skip("PyTorch-specific fractional_matrix_power regression test")

    values = backend.asarray(
        [
            [
                [[4.0, 0.0], [0.0, 9.0]],
                [[16.0, 0.0], [0.0, 25.0]],
                [[36.0, 0.0], [0.0, 49.0]],
            ],
            [
                [[64.0, 0.0], [0.0, 81.0]],
                [[100.0, 0.0], [0.0, 121.0]],
                [[144.0, 0.0], [0.0, 169.0]],
            ],
        ]
    )

    result = linalg.fractional_matrix_power(values, 0.5)

    expected = backend.asarray(
        [
            [
                [[2.0, 0.0], [0.0, 3.0]],
                [[4.0, 0.0], [0.0, 5.0]],
                [[6.0, 0.0], [0.0, 7.0]],
            ],
            [
                [[8.0, 0.0], [0.0, 9.0]],
                [[10.0, 0.0], [0.0, 11.0]],
                [[12.0, 0.0], [0.0, 13.0]],
            ],
        ]
    )

    assert result.shape == (2, 3, 2, 2)
    assert bool(_to_python(backend.allclose(result, expected)))
