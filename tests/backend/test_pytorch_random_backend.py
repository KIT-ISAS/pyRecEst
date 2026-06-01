import pytest

pytest.importorskip("torch")

from pyrecest._backend.pytorch import random  # noqa: E402


@pytest.mark.parametrize(
    "bad_size",
    [True, False, (True,), [False, 2], 1.5, (2.0,), "3"],
)
def test_size_arguments_reject_bool_and_non_integral_dimensions(bad_size):
    samplers = (
        lambda size: random.rand(size=size),
        lambda size: random.uniform(size=size),
        lambda size: random.normal(size=size),
        lambda size: random.randint(0, 3, size=size),
        lambda size: random.choice(3, size=size),
        lambda size: random.multivariate_normal([0.0], [[1.0]], size=size),
    )

    for sampler in samplers:
        with pytest.raises(TypeError):
            sampler(bad_size)


@pytest.mark.parametrize("bad_size", [-1, (2, -1)])
def test_size_arguments_reject_negative_dimensions(bad_size):
    samplers = (
        lambda size: random.rand(size=size),
        lambda size: random.uniform(size=size),
        lambda size: random.normal(size=size),
        lambda size: random.randint(0, 3, size=size),
        lambda size: random.choice(3, size=size),
        lambda size: random.multivariate_normal([0.0], [[1.0]], size=size),
    )

    for sampler in samplers:
        with pytest.raises(ValueError):
            sampler(bad_size)


def test_scalar_and_empty_tuple_sizes_keep_scalar_shape():
    assert random.rand().shape == ()
    assert random.rand(size=()).shape == ()
    assert random.normal(size=()).shape == ()
    assert random.uniform(size=()).shape == ()
    assert random.randint(0, 3, size=()).shape == ()
    assert random.multivariate_normal([0.0], [[1.0]], size=()).shape == (1,)


def test_zero_sized_choice_still_works_for_empty_population():
    sample = random.choice(0, size=(0,))

    assert sample.shape == (0,)
