import numpy as np
import pyrecest.backend as backend
import pytest


def _as_numpy(value):
    return backend.to_numpy(value)


@pytest.mark.parametrize("axis", [0, 1, 2, -1])
def test_pytorch_apply_along_axis_matches_numpy_for_scalar_results(axis):
    if backend.__backend_name__ != "pytorch":
        pytest.skip("PyTorch-specific apply_along_axis backend contract")

    values_np = np.arange(24.0).reshape(2, 3, 4)
    values = backend.asarray(values_np)

    actual = _as_numpy(backend.apply_along_axis(backend.sum, axis, values))
    expected = np.apply_along_axis(np.sum, axis, values_np)

    assert actual.shape == expected.shape
    assert np.allclose(actual, expected)


@pytest.mark.parametrize("axis", [0, 1, 2, -1])
def test_pytorch_apply_along_axis_matches_numpy_for_vector_results(axis):
    if backend.__backend_name__ != "pytorch":
        pytest.skip("PyTorch-specific apply_along_axis backend contract")

    values_np = np.arange(24.0).reshape(2, 3, 4)
    values = backend.asarray(values_np)

    actual = _as_numpy(
        backend.apply_along_axis(lambda row: row[:2] * 2.0, axis, values)
    )
    expected = np.apply_along_axis(lambda row: row[:2] * 2.0, axis, values_np)

    assert actual.shape == expected.shape
    assert np.allclose(actual, expected)


def test_pytorch_apply_along_axis_forwards_callback_arguments():
    if backend.__backend_name__ != "pytorch":
        pytest.skip("PyTorch-specific apply_along_axis backend contract")

    values_np = np.arange(12.0).reshape(3, 4)
    values = backend.asarray(values_np)

    def affine_prefix(row, scale, offset=0.0):
        return row[:2] * scale + offset

    actual = _as_numpy(
        backend.apply_along_axis(affine_prefix, 1, values, 2.0, offset=1.5)
    )
    expected = np.apply_along_axis(affine_prefix, 1, values_np, 2.0, offset=1.5)

    assert actual.shape == expected.shape
    assert np.allclose(actual, expected)
