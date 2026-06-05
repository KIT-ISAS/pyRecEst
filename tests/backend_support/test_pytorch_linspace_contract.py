import numpy as np
import pyrecest.backend as backend
import pytest


def _as_numpy(value):
    return backend.to_numpy(value)


@pytest.mark.parametrize("endpoint", [True, False])
@pytest.mark.parametrize(
    ("start_np", "stop_np"),
    [
        (0.0, np.array([2.0, 3.0])),
        (np.array([0.0, 1.0]), 1.0),
        (np.array([0.0, 1.0]), np.array([2.0, 3.0])),
    ],
)
def test_pytorch_linspace_matches_numpy_for_array_like_endpoints(
    start_np, stop_np, endpoint
):
    if backend.__backend_name__ != "pytorch":
        pytest.skip("PyTorch-specific linspace backend contract")

    actual = _as_numpy(backend.linspace(start_np, stop_np, num=5, endpoint=endpoint))
    expected = np.linspace(start_np, stop_np, num=5, endpoint=endpoint)

    assert actual.shape == expected.shape
    assert np.allclose(actual, expected)


def test_pytorch_linspace_endpoint_false_allows_constant_lanes():
    if backend.__backend_name__ != "pytorch":
        pytest.skip("PyTorch-specific linspace backend contract")

    start_np = np.array([0.0, 1.0])
    stop_np = np.array([2.0, 1.0])

    actual = _as_numpy(
        backend.linspace(backend.asarray(start_np), stop_np, num=5, endpoint=False)
    )
    expected = np.linspace(start_np, stop_np, num=5, endpoint=False)

    assert actual.shape == expected.shape
    assert np.allclose(actual, expected)


def test_pytorch_linspace_preserves_explicit_integer_dtype():
    if backend.__backend_name__ != "pytorch":
        pytest.skip("PyTorch-specific linspace backend contract")

    actual = _as_numpy(backend.linspace([0, 1], [2, 3], num=3, dtype=backend.int64))
    expected = np.linspace([0, 1], [2, 3], num=3, dtype=np.int64)

    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert np.array_equal(actual, expected)
