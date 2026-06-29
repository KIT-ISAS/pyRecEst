import importlib.util

import pytest


def _to_python(sort_backend, value):
    value = sort_backend.to_numpy(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


@pytest.mark.backend_portable
def test_pytorch_sort_accepts_numpy_axis_none_and_stable_kind():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    import pyrecest  # noqa: F401  # pylint: disable=import-outside-toplevel,unused-import
    import pyrecest.backend as public_backend  # pylint: disable=import-outside-toplevel
    import pyrecest._backend.pytorch as sort_backend  # pylint: disable=import-outside-toplevel

    assert _to_python(sort_backend, sort_backend.sort([[3, 1], [2, 4]], axis=None)) == [1, 2, 3, 4]
    assert _to_python(sort_backend, sort_backend.sort([3, 1, 2, 1], kind="stable")) == [1, 1, 2, 3]
    assert _to_python(sort_backend, sort_backend.sort([3, 1, 2, 1], kind="mergesort")) == [1, 1, 2, 3]

    if public_backend.__backend_name__ == "pytorch":
        assert _to_python(public_backend, public_backend.sort([[3, 1], [2, 4]], axis=None)) == [1, 2, 3, 4]
        assert _to_python(public_backend, public_backend.sort([3, 1, 2, 1], kind="stable")) == [1, 1, 2, 3]
