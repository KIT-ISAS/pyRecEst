import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
from pyrecest._backend.jax import random  # noqa: E402


def test_multivariate_normal_accepts_numpy_argument_order():
    random.seed(0)
    mean = jnp.array([1.0, -1.0])
    cov = jnp.eye(2)

    assert random.multivariate_normal(mean, cov).shape == (2,)
    assert random.multivariate_normal(mean, cov, 3).shape == (3, 2)
    assert random.multivariate_normal(mean, cov, (4,)).shape == (4, 2)


def test_multivariate_normal_accepts_shape_keyword():
    random.seed(0)
    mean = jnp.array([1.0, -1.0])
    cov = jnp.eye(2)

    assert random.multivariate_normal(mean, cov, shape=(5,)).shape == (5, 2)

    with pytest.raises(TypeError):
        random.multivariate_normal(mean, cov, size=(1,), shape=(2,))


def test_uniform_accepts_numpy_broadcasted_bounds_without_explicit_size():
    random.seed(0)

    samples = random.uniform(jnp.array([0.0, 10.0]), jnp.array([1.0, 11.0]))

    assert samples.shape == (2,)
    assert jnp.all(samples >= jnp.array([0.0, 10.0]))
    assert jnp.all(samples <= jnp.array([1.0, 11.0]))


def test_randint_accepts_numpy_broadcasted_bounds_without_explicit_size():
    random.seed(0)

    samples = random.randint(jnp.array([0, 10]), jnp.array([3, 13]))

    assert samples.shape == (2,)
    assert jnp.all(samples >= jnp.array([0, 10]))
    assert jnp.all(samples < jnp.array([3, 13]))


def test_uniform_and_randint_reject_incompatible_bounds_without_explicit_size():
    with pytest.raises(ValueError):
        random.uniform(jnp.zeros((2,)), jnp.ones((3,)))

    with pytest.raises(ValueError):
        random.randint(
            jnp.zeros((2,), dtype=jnp.int32), jnp.ones((3,), dtype=jnp.int32)
        )


def _size_aware_samplers():
    values = jnp.array([0, 1, 2])
    mean = jnp.array([0.0])
    cov = jnp.eye(1)
    return (
        lambda size: random.rand(size=size),
        lambda size: random.uniform(size=size),
        lambda size: random.normal(size=size),
        lambda size: random.randint(0, 3, size=size),
        lambda size: random.choice(values, size=size),
        lambda size: random.multivariate_normal(mean, cov, size=size),
        lambda size: random.multinomial(3, [0.25, 0.75], size=size),
    )


@pytest.mark.parametrize(
    "bad_size",
    [True, False, np.bool_(True), (True,), [np.bool_(False), 2], 1.5, (2.0,), "3"],
)
def test_size_arguments_reject_bool_and_non_integral_dimensions(bad_size):
    for sampler in _size_aware_samplers():
        with pytest.raises(TypeError):
            sampler(bad_size)


@pytest.mark.parametrize("bad_size", [-1, (2, -1)])
def test_size_arguments_reject_negative_dimensions(bad_size):
    for sampler in _size_aware_samplers():
        with pytest.raises(ValueError):
            sampler(bad_size)


def test_normal_legacy_shape_detection_accepts_numpy_integer_dimensions():
    random.seed(0)

    assert random.normal(np.int64(3)).shape == (3,)
    assert random.normal((np.int64(2), 3)).shape == (2, 3)


def test_normal_bool_location_is_not_interpreted_as_legacy_shape():
    random.seed(0)

    assert random.normal(True).shape == ()


def test_normal_integer_tuple_location_with_array_scale_is_not_legacy_shape():
    random.seed(0)

    sample = random.normal((1, 2), scale=jnp.ones(2))

    assert sample.shape == (2,)


def test_normal_accepts_numpy_broadcasted_parameters_without_explicit_size():
    random.seed(0)

    sample = random.normal(jnp.array([0.0, 0.0]), jnp.ones(2))

    assert sample.shape == (2,)
    assert sample[0] != sample[1]


def test_normal_rejects_negative_scale():
    with pytest.raises(ValueError, match="non-negative"):
        random.normal(scale=-1.0)

    with pytest.raises(ValueError, match="non-negative"):
        random.normal(scale=jnp.array([1.0, -0.1]))


def test_multinomial_accepts_numpy_size_argument():
    random.seed(0)

    samples = random.multinomial(12, jnp.array([0.25, 0.75]), size=(2, 3))

    assert samples.shape == (2, 3, 2)
    assert jnp.all(jnp.sum(samples, axis=-1) == 12)


def test_multinomial_explicit_state_with_size_does_not_mutate_global_state():
    state = random.create_random_state(123)
    original_global_state = random.get_state()

    state_after, samples = random.multinomial(5, [0.25, 0.75], size=4, state=state)

    assert samples.shape == (4, 2)
    assert jnp.all(jnp.sum(samples, axis=-1) == 5)
    assert jnp.array_equal(random.get_state(), original_global_state)
    assert not jnp.array_equal(state, state_after)
