import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from pyrecest._backend.jax import autodiff  # noqa: E402


def test_custom_gradient_contracts_full_jacobian_for_vector_output():
    @autodiff.custom_gradient(lambda x: jnp.diag(jnp.array([2.0, 3.0])))
    def linear_map(x):
        return jnp.array([2.0 * x[0], 3.0 * x[1]])

    result = jax.grad(lambda x: jnp.sum(linear_map(x)))(jnp.array([5.0, 7.0]))

    assert jnp.allclose(result, jnp.array([2.0, 3.0]))


def test_custom_gradient_contracts_vector_derivative_for_scalar_argument():
    @autodiff.custom_gradient(lambda x: jnp.array([2.0, 3.0]))
    def vector_from_scalar(x):
        return jnp.array([2.0 * x, 3.0 * x])

    result = jax.grad(lambda x: jnp.sum(vector_from_scalar(x)))(4.0)

    assert jnp.allclose(result, 5.0)
