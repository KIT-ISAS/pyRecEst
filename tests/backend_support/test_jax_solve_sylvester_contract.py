import pyrecest.backend as backend
import pytest
from pyrecest._backend.capabilities import get_unsupported_functions
from pyrecest.backend import array, linalg


def _to_python(value):
    value = backend.to_numpy(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def test_jax_solve_sylvester_is_not_declared_unsupported():
    assert "solve_sylvester" not in get_unsupported_functions("jax", "linalg")


def test_jax_solve_sylvester_solves_sylvester_equation():
    if backend.__backend_name__ != "jax":
        pytest.skip("JAX-specific linalg regression test")

    a = array([[2.0, 0.0], [0.0, 3.0]])
    b = array([[5.0, 0.0], [0.0, 7.0]])
    q = array([[1.0, 2.0], [3.0, 4.0]])

    x = linalg.solve_sylvester(a, b, q)
    residual = backend.matmul(a, x) + backend.matmul(x, b)

    assert tuple(x.shape) == (2, 2)
    assert bool(_to_python(backend.allclose(residual, q, atol=1e-6)))
