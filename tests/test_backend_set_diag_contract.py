import pyrecest.backend as backend


def _to_python(value):
    value = backend.to_numpy(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def test_set_diag_accepts_array_like_matrix_inputs():
    result = backend.set_diag([[1, 2], [3, 4]], [9, 8])

    assert _to_python(result) == [[9, 2], [3, 8]]
