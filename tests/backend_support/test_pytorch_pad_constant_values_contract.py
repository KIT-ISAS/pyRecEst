import pytest

from tests.support.backend_runner import run_backend_code


def test_pytorch_pad_accepts_numpy_constant_value_pairs():
    pytest.importorskip("torch")

    code = """
import numpy as np
import numpy.testing as npt

import pyrecest.backend as backend

values = backend.asarray([[1, 2], [3, 4]], dtype=backend.int64)
result = backend.pad(values, ((1, 0), (0, 2)), constant_values=((5, 6), (7, 8)))
expected = np.pad(
    np.array([[1, 2], [3, 4]]),
    ((1, 0), (0, 2)),
    mode="constant",
    constant_values=((5, 6), (7, 8)),
)
assert result.shape == expected.shape
assert backend.to_numpy(result).tolist() == expected.tolist()

complex_values = backend.asarray([1.0 + 1.0j], dtype=backend.complex128)
complex_result = backend.pad(complex_values, (1, 1), constant_values=2.0 + 3.0j)
complex_expected = np.pad(
    np.array([1.0 + 1.0j], dtype=np.complex128),
    (1, 1),
    mode="constant",
    constant_values=2.0 + 3.0j,
)
npt.assert_allclose(backend.to_numpy(complex_result), complex_expected)
"""

    result = run_backend_code("pytorch", code)

    assert result.returncode == 0, result.stderr
