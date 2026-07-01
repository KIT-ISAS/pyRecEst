"""Regression tests for PyTorch norm axis normalization."""

import numpy as np
import pytest


def test_pytorch_norm_accepts_numpy_array_axes():
    torch = pytest.importorskip("torch")
    from pyrecest._backend.pytorch import linalg

    x = torch.arange(1, 7, dtype=torch.float64).reshape(2, 3)

    scalar_axis = linalg.norm(x, axis=np.array(1))
    expected_scalar_axis = torch.linalg.norm(x, dim=1)
    torch.testing.assert_close(scalar_axis, expected_scalar_axis)

    singleton_axis = linalg.norm(x, axis=np.array([0]), keepdims=True)
    expected_singleton_axis = torch.linalg.norm(x, dim=(0,), keepdim=True)
    torch.testing.assert_close(singleton_axis, expected_singleton_axis)

    matrix_axis = linalg.norm(x, ord="fro", axis=np.array([0, 1]))
    expected_matrix_axis = torch.linalg.norm(x, ord="fro", dim=(0, 1))
    torch.testing.assert_close(matrix_axis, expected_matrix_axis)
