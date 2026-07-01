import importlib.util

import numpy as np
import numpy.testing as npt
import pytest


CASES = [
    ([1.0, 2.0], [[5.0, 6.0], [7.0, 8.0]]),
    ([[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]),
    (2.0, [[5.0, 6.0], [7.0, 8.0]]),
    (
        np.arange(24, dtype=float).reshape(2, 3, 4),
        np.arange(120, dtype=float).reshape(5, 4, 6),
    ),
]


def _assert_dot_matches_numpy(raw_pytorch, left, right):
    result = raw_pytorch.dot(raw_pytorch.array(left), raw_pytorch.array(right))
    expected = np.dot(np.asarray(left), np.asarray(right))
    npt.assert_allclose(raw_pytorch.to_numpy(result), expected)
    assert result.shape == expected.shape


@pytest.mark.backend_portable
def test_raw_pytorch_dot_matches_numpy_contract():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    import pyrecest._backend.pytorch as raw_pytorch

    for left, right in CASES:
        _assert_dot_matches_numpy(raw_pytorch, left, right)
