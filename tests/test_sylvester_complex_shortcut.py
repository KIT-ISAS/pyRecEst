import importlib.util

import numpy as np
import numpy.testing as npt
import pytest
from pyrecest._backend import numpy as numpy_backend

pytorch_backend = None
if importlib.util.find_spec("torch") is not None:
    from pyrecest._backend import pytorch as pytorch_backend


def test_numpy_solve_sylvester_falls_back_for_complex_symmetric_nonhermitian():
    a = np.array([[2.0 + 0.0j, 1.0j], [1.0j, 2.0 + 0.0j]], dtype=np.complex128)
    q = np.eye(2, dtype=np.complex128)

    x = numpy_backend.linalg.solve_sylvester(a, a, q)

    residual = a @ x + x @ a - q
    npt.assert_allclose(residual, np.zeros_like(q), atol=1e-12)


def test_numpy_solve_sylvester_handles_complex_hermitian_shortcut():
    a = np.array([[2.0 + 0.0j, 1.0j], [-1.0j, 2.0 + 0.0j]], dtype=np.complex128)
    q = np.eye(2, dtype=np.complex128)

    x = numpy_backend.linalg.solve_sylvester(a, a, q)

    residual = a @ x + x @ a - q
    npt.assert_allclose(residual, np.zeros_like(q), atol=1e-12)


@pytest.mark.skipif(pytorch_backend is None, reason="PyTorch is not installed")
def test_pytorch_solve_sylvester_handles_complex_hermitian_general_path():
    a = pytorch_backend.array(
        [[2.0 + 0.0j, 1.0j], [-1.0j, 2.0 + 0.0j]],
        dtype=pytorch_backend.complex128,
    )
    q = pytorch_backend.array(
        [[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 1.0 + 0.0j]],
        dtype=pytorch_backend.complex128,
    )

    x = pytorch_backend.linalg.solve_sylvester(a, a, q)

    residual = a @ x + x @ a - q
    residual_norm = pytorch_backend.linalg.norm(residual)
    assert float(pytorch_backend.to_numpy(residual_norm)) < 1e-12
