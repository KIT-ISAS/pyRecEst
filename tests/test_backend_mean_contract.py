import pyrecest.backend as backend


def _to_python(value):
    value = backend.to_numpy(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def test_mean_accepts_axis_and_keepdims_keywords():
    values = backend.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    result = backend.mean(values, axis=1, keepdims=True)

    assert _to_python(result) == [[2.0], [5.0]]


def test_mean_accepts_tuple_axis():
    values = backend.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ]
    )

    result = backend.mean(values, axis=(0, 2))

    assert _to_python(result) == [3.5, 5.5]
