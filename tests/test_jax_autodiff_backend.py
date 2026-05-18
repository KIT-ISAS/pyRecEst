import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from pyrecest._backend.jax import autodiff


def test_elementwise_grad_vector_output_returns_elementwise_derivative():
    grad_fn = autodiff.elementwise_grad(lambda x: x**2)

    result = grad_fn(jnp.array([1.0, 2.0, -3.0]))

    assert jnp.allclose(result, jnp.array([2.0, 4.0, -6.0]))


def test_elementwise_grad_scalar_output_returns_regular_gradient():
    grad_fn = autodiff.elementwise_grad(lambda x: jnp.sum(x**2))

    result = grad_fn(jnp.array([1.0, 2.0, -3.0]))

    assert jnp.allclose(result, jnp.array([2.0, 4.0, -6.0]))


@pytest.mark.parametrize(
    "function_name",
    [
        "hessian",
        "hessian_vec",
        "jacobian_vec",
        "jacobian_and_hessian",
        "value_jacobian_and_hessian",
        "value_and_jacobian",
    ],
)
def test_unsupported_autodiff_functions_raise_not_implemented(function_name):
    with pytest.raises(NotImplementedError, match="not supported in this JAX backend"):
        getattr(autodiff, function_name)(lambda x: x)
