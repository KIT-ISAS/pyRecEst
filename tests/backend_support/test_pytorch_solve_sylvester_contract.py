import numpy as np
import pyrecest.backend as backend
import pytest
from scipy.linalg import solve_sylvester as scipy_solve_sylvester


def test_pytorch_solve_sylvester_general_case_matches_scipy():
    if backend.__backend_name__ != "pytorch":
        pytest.skip("PyTorch-specific Sylvester solver contract")

    a_np = np.array([[1.0, 2.0], [0.0, 3.0]])
    b_np = np.array([[0.5, -1.0], [2.0, 1.5]])
    q_np = np.array([[1.0, 0.5], [-0.25, 2.0]])

    result = backend.linalg.solve_sylvester(
        backend.asarray(a_np),
        backend.asarray(b_np),
        backend.asarray(q_np),
    )

    assert result is not None
    actual = backend.to_numpy(result)
    expected = scipy_solve_sylvester(a_np, b_np, q_np)

    assert actual.shape == expected.shape
    assert np.allclose(actual, expected)


def test_pytorch_solve_sylvester_general_case_supports_batches():
    if backend.__backend_name__ != "pytorch":
        pytest.skip("PyTorch-specific Sylvester solver contract")

    a_np = np.array(
        [
            [[1.0, 2.0], [0.0, 3.0]],
            [[2.0, -0.5], [1.0, 4.0]],
        ]
    )
    b_np = np.array(
        [
            [[0.5, -1.0], [2.0, 1.5]],
            [[1.0, 0.25], [-0.75, 2.0]],
        ]
    )
    q_np = np.array(
        [
            [[1.0, 0.5], [-0.25, 2.0]],
            [[0.25, -1.0], [1.5, 0.75]],
        ]
    )

    result = backend.linalg.solve_sylvester(
        backend.asarray(a_np),
        backend.asarray(b_np),
        backend.asarray(q_np),
    )

    assert result is not None
    actual = backend.to_numpy(result)
    expected = np.stack(
        [
            scipy_solve_sylvester(a, b, q)
            for a, b, q in zip(a_np, b_np, q_np)
        ]
    )

    assert actual.shape == expected.shape
    assert np.allclose(actual, expected)
