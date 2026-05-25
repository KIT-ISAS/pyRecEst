import importlib.util
import os
import subprocess
import sys

import pyrecest.backend as backend
import pytest


def _to_python(value):
    value = backend.to_numpy(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def test_sum_accepts_axis_and_keepdims_keywords():
    values = backend.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    result = backend.sum(values, axis=1, keepdims=True)

    assert _to_python(result) == [[6.0], [15.0]]


def test_sum_accepts_axis_none_with_keepdims():
    values = backend.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    result = backend.sum(values, keepdims=True)

    assert _to_python(result) == [[21.0]]


def test_sum_accepts_tuple_axis():
    values = backend.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ]
    )

    result = backend.sum(values, axis=(0, 2))

    assert _to_python(result) == [14.0, 22.0]


@pytest.mark.backend_portable
def test_pytorch_sum_accepts_axis_none_with_keepdims():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    env = os.environ.copy()
    env["PYRECEST_BACKEND"] = "pytorch"
    src_path = os.path.abspath("src")
    env["PYTHONPATH"] = (
        src_path
        if not env.get("PYTHONPATH")
        else os.pathsep.join([src_path, env["PYTHONPATH"]])
    )

    code = """
import pyrecest.backend as backend
values = backend.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
total = backend.sum(values, keepdims=True)
assert backend.to_numpy(total).tolist() == [[21.0]]
row_sums = backend.sum(values, axis=1, keepdims=True)
assert backend.to_numpy(row_sums).tolist() == [[6.0], [15.0]]
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
