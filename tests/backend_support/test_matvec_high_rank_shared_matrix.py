import pyrecest.backend as backend
from pyrecest.backend import array


def _to_python(value):
    value = backend.to_numpy(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def test_matvec_shared_matrix_accepts_asymmetric_high_rank_vector_batches():
    matrix = array([[1.0, 2.0], [3.0, 4.0]])
    vectors = array(
        [
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
                [[9.0, 10.0], [11.0, 12.0]],
            ],
            [
                [[13.0, 14.0], [15.0, 16.0]],
                [[17.0, 18.0], [19.0, 20.0]],
                [[21.0, 22.0], [23.0, 24.0]],
            ],
            [
                [[25.0, 26.0], [27.0, 28.0]],
                [[29.0, 30.0], [31.0, 32.0]],
                [[33.0, 34.0], [35.0, 36.0]],
            ],
            [
                [[37.0, 38.0], [39.0, 40.0]],
                [[41.0, 42.0], [43.0, 44.0]],
                [[45.0, 46.0], [47.0, 48.0]],
            ],
        ]
    )

    result = backend.matvec(matrix, vectors)

    assert result.shape == (4, 3, 2, 2)
    assert _to_python(result[0, 0, 0]) == [5.0, 11.0]
    assert _to_python(result[3, 2, 1]) == [143.0, 333.0]
