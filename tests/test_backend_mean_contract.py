import importlib.util

import pyrecest.backend as backend
import pytest
from tests.support.backend_runner import run_backend_code


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


@pytest.mark.backend_portable
def test_pytorch_mean_accepts_integer_array_like_inputs():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        "pytorch",
        """
import pyrecest.backend as backend

result = backend.mean([[1, 2, 3], [4, 5, 6]], axis=0, keepdims=True)
expected = backend.array([[2.5, 3.5, 4.5]])
assert tuple(result.shape) == (1, 3)
assert result.dtype == backend.float64
assert backend.allclose(result, expected)
print("ok")
""",
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
