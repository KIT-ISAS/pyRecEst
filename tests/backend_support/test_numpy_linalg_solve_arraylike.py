"""Regression coverage for shared NumPy linalg.solve array-like inputs."""

from pyrecest._backend._shared_numpy import linalg


def test_solve_accepts_array_like_vector_rhs():
    solution = linalg.solve([[2.0, 0.0], [0.0, 4.0]], [2.0, 8.0])

    assert solution.shape == (2,)
    assert solution.tolist() == [1.0, 2.0]


def test_solve_accepts_array_like_matrix_rhs():
    solution = linalg.solve(
        [[2.0, 0.0], [0.0, 4.0]],
        [[2.0, 4.0], [8.0, 12.0]],
    )

    assert solution.shape == (2, 2)
    assert solution.tolist() == [[1.0, 2.0], [2.0, 3.0]]
